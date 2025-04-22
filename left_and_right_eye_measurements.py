import cv2
import dlib
import numpy as np
import time
from datetime import datetime
import pandas as pd
import os
import torch
import glob

# Add imports for SAM model
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    print("Segment Anything Model (SAM) not available. Millimeter calibration will not be used.")
    SAM_AVAILABLE = False

class EyeTracker:
    def __init__(self, file=0, saving=False, testing=False, use_calibration=True):
        # save constructor inputs
        self.testing = testing
        self.saving = saving
        self.file = file
        self.use_calibration = use_calibration and SAM_AVAILABLE
        
        # Initialize face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Initialize eye detector using Haar cascades - use local file first
        if os.path.exists('haarcascade_eye.xml'):
            self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
            if self.testing:
                print("Using local haarcascade_eye.xml file")
        else:
            # Try standard locations as fallback
            cascade_path = cv2.data.haarcascades if hasattr(cv2, 'data') else '/usr/local/share/opencv4/haarcascades/'
            eye_cascade_path = os.path.join(cascade_path, 'haarcascade_eye.xml')
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            
            # Fallback to absolute path if the first method doesn't work
            if self.eye_cascade.empty():
                # Try finding the file in common OpenCV installation locations
                common_paths = [
                    '/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml',
                    '/usr/share/opencv4/haarcascades/haarcascade_eye.xml',
                    '/usr/local/share/opencv/haarcascades/haarcascade_eye.xml',
                    '/usr/share/opencv/haarcascades/haarcascade_eye.xml'
                ]
                
                for path in common_paths:
                    if os.path.exists(path):
                        self.eye_cascade = cv2.CascadeClassifier(path)
                        if not self.eye_cascade.empty():
                            break
                
                # Still empty? Try to download it
                if self.eye_cascade.empty():
                    print("Could not find haarcascade_eye.xml. Attempting to download...")
                    import urllib.request
                    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
                    urllib.request.urlretrieve(url, "haarcascade_eye.xml")
                    self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
                
                if self.eye_cascade.empty():
                    print("ERROR: Could not load eye cascade classifier. Eye detection will not work.")
        
        # Initialize video capture
        if file != 0 and isinstance(file, str) and os.path.exists(file):
            self.cap = cv2.VideoCapture(file)
        else:
            self.cap = cv2.VideoCapture(file)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Storage for measurements
        self.measurements = []
        self.left_eye_frames = list()
        self.right_eye_frames = list()
        
        # Create a directory to save frames and measurements
        self.output_dir = "eye_tracking_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Calibration parameters
        self.is_calibrated = False
        self.left_pixels_per_mm = None
        self.right_pixels_per_mm = None
        
        # Average corneal diameter (iris) in millimeters - standard medical value
        self.CORNEAL_DIAMETER_MM = 11.8
        
        # Initialize SAM model if available and calibration is requested
        if self.use_calibration:
            try:
                # SAM model setup (will be initialized only when needed)
                self.sam_model = None
                self.sam_predictor = None  # Renamed from self.predictor to avoid confusion with dlib predictor
                print("SAM calibration will be used for millimeter measurements")
            except Exception as e:
                print(f"Error initializing SAM model: {e}")
                self.use_calibration = False
        
    def initialize_sam_model(self):
        """Initialize the SAM model when needed"""
        if not self.use_calibration or not SAM_AVAILABLE:
            return False
            
        try:
            # Look for model weights in the current directory first, then in models/
            model_path = None
            # Check current directory for .pth files that contain "sam"
            current_dir_models = glob.glob("*.pth")
            sam_models = [m for m in current_dir_models if "sam" in m.lower()]
            
            if sam_models:
                model_path = sam_models[0]
                print(f"Found SAM model in current directory: {model_path}")
            else:
                # Check models directory as fallback
                models_dir_path = "models/sam_vit_b_01ec64.pth"
                if os.path.exists(models_dir_path):
                    model_path = models_dir_path
                    
            if not model_path:
                print(f"SAM model weights not found. Please ensure a .pth file with 'sam' in the name exists in the current directory")
                self.use_calibration = False
                return False
            
            # Determine the model type based on the filename
            model_type = "vit_b"  # Default model type
            if "vit_h" in model_path.lower():
                model_type = "vit_h"
            elif "vit_l" in model_path.lower():
                model_type = "vit_l"
                
            # Initialize the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device=device)
            self.sam_predictor = SamPredictor(sam)
            print(f"SAM model initialized (type: {model_type}) on {device}")
            return True
        except Exception as e:
            print(f"Failed to initialize SAM model: {e}")
            self.use_calibration = False
            return False
    
    def detect_iris(self, eye_image):
        """Use SAM to detect the iris in the eye image and calculate its diameter"""
        if not self.sam_predictor:
            if not self.initialize_sam_model():
                return None
                
        try:
            # Convert BGR to RGB for SAM
            rgb_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
            
            # Set the image for the predictor
            self.sam_predictor.set_image(rgb_image)
            
            # Generate a center point for the iris (approximate center of the image)
            h, w = eye_image.shape[:2]
            center_point = np.array([[w//2, h//2]])
            
            # Get segmentation mask using the center point
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=center_point,
                point_labels=np.array([1]),  # 1 for foreground
                multimask_output=True
            )
            
            # Find the mask with the highest score
            best_mask_idx = np.argmax(scores)
            iris_mask = masks[best_mask_idx]
            
            # Extract contours from the mask
            iris_mask = iris_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If no contours found, return None
            if not contours:
                return None
                
            # Find the largest contour (should be the iris)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Fit a circle to the contour
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            diameter = radius * 2
            
            if self.testing:
                # Draw the iris contour and circle for visualization
                vis_image = eye_image.copy()
                cv2.drawContours(vis_image, [largest_contour], 0, (0, 255, 0), 2)
                cv2.circle(vis_image, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                cv2.imwrite(os.path.join(self.output_dir, "iris_detection.jpg"), vis_image)
                print(f"Iris diameter in pixels: {diameter}")
                
            return diameter
        except Exception as e:
            print(f"Error in iris detection: {e}")
            return None
            
    def calibrate(self):
        """Calibrate the pixel to millimeter conversion using the corneal diameter"""
        if not self.use_calibration:
            return False
            
        try:
            # Capture a frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame for calibration")
                return False
                
            # Get eye measurements and regions
            frame, left_measurements, right_measurements = self.get_eye_measurements(frame, for_calibration=True)
            
            # If no faces or eyes detected, return False
            if not self.left_eye_frames or not self.right_eye_frames:
                print("No eyes detected for calibration")
                return False
                
            # Use the last frame from each eye
            left_eye = self.left_eye_frames[-1]
            right_eye = self.right_eye_frames[-1]
            
            # Detect iris diameter in pixels
            left_corneal_diameter_px = self.detect_iris(left_eye)
            right_corneal_diameter_px = self.detect_iris(right_eye)
            
            if left_corneal_diameter_px and right_corneal_diameter_px:
                # Calculate pixels per millimeter
                self.left_pixels_per_mm = left_corneal_diameter_px / self.CORNEAL_DIAMETER_MM
                self.right_pixels_per_mm = right_corneal_diameter_px / self.CORNEAL_DIAMETER_MM
                
                if self.testing:
                    print(f"Calibration complete: Left eye {self.left_pixels_per_mm:.2f} px/mm, Right eye {self.right_pixels_per_mm:.2f} px/mm")
                
                self.is_calibrated = True
                return True
            else:
                print("Failed to detect iris for calibration")
                return False
                
        except Exception as e:
            print(f"Calibration error: {e}")
            return False
        
    def get_output_dir(self):
        """return the output directory containing the images"""
        return self.output_dir
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_eye_measurements(self, frame, for_calibration=False):
        """Extract eye measurements from a single frame using facial landmarks"""
        original_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Always attempt facial landmark detection first
        faces = self.detector(gray)
        
        # Variables to store detection results
        left_eye_x, left_eye_y, left_eye_w, left_eye_h = 0, 0, 0, 0
        right_eye_x, right_eye_y, right_eye_w, right_eye_h = 0, 0, 0, 0
        left_eye_image = None
        right_eye_image = None
        
        # Eye landmark points
        left_upper_lid = None
        left_lower_lid = None
        left_pupil_center = None
        right_upper_lid = None
        right_lower_lid = None
        right_pupil_center = None
        
        if len(faces) > 0:
            # Use facial landmarks for more accurate eye detection
            if self.testing:
                print(f"Detected {len(faces)} face(s), using facial landmarks for precise eye measurements")
                
            # Process the first face detected
            face = faces[0]
            
            # Get facial landmarks
            landmarks = self.predictor(gray, face)
            
            # Define eye landmark indices in the 68-point model:
            # Left eye: points 36-41
            # Right eye: points 42-47
            left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            
            # Calculate eye regions
            left_eye_x = min(p[0] for p in left_eye_points)
            left_eye_y = min(p[1] for p in left_eye_points)
            left_eye_w = max(p[0] for p in left_eye_points) - left_eye_x
            left_eye_h = max(p[1] for p in left_eye_points) - left_eye_y
            
            right_eye_x = min(p[0] for p in right_eye_points)
            right_eye_y = min(p[1] for p in right_eye_points)
            right_eye_w = max(p[0] for p in right_eye_points) - right_eye_x
            right_eye_h = max(p[1] for p in right_eye_points) - right_eye_y
            
            # Add padding around eyes
            padding = 10
            left_eye_x = max(0, left_eye_x - padding)
            left_eye_y = max(0, left_eye_y - padding)
            left_eye_w += 2 * padding
            left_eye_h += 2 * padding
            
            right_eye_x = max(0, right_eye_x - padding)
            right_eye_y = max(0, right_eye_y - padding)
            right_eye_w += 2 * padding
            right_eye_h += 2 * padding
            
            # Ensure we don't go beyond image boundaries
            h, w = frame.shape[:2]
            left_eye_w = min(left_eye_w, w - left_eye_x)
            left_eye_h = min(left_eye_h, h - left_eye_y)
            right_eye_w = min(right_eye_w, w - right_eye_x)
            right_eye_h = min(right_eye_h, h - right_eye_y)
            
            # Extract eye images
            left_eye_image = original_frame[left_eye_y:left_eye_y+left_eye_h, left_eye_x:left_eye_x+left_eye_w]
            right_eye_image = original_frame[right_eye_y:right_eye_y+right_eye_h, right_eye_x:right_eye_x+right_eye_w]
            
            # Get precise lid landmarks using facial landmarks
            # For left eye - upper lid is midpoint between points 37 and 38
            left_upper_lid = ((landmarks.part(37).x + landmarks.part(38).x) // 2, 
                            (landmarks.part(37).y + landmarks.part(38).y) // 2)
            # For left eye - lower lid is midpoint between points 40 and 41
            left_lower_lid = ((landmarks.part(40).x + landmarks.part(41).x) // 2, 
                            (landmarks.part(40).y + landmarks.part(41).y) // 2)
            
            # For right eye - upper lid is midpoint between points 43 and 44
            right_upper_lid = ((landmarks.part(43).x + landmarks.part(44).x) // 2, 
                             (landmarks.part(43).y + landmarks.part(44).y) // 2)
            # For right eye - lower lid is midpoint between points 46 and 47
            right_lower_lid = ((landmarks.part(46).x + landmarks.part(47).x) // 2, 
                             (landmarks.part(46).y + landmarks.part(47).y) // 2)
            
        else:
            # Fallback to direct eye detection if no face detected
            if self.testing:
                print("No faces detected, falling back to Haar cascade for direct eye detection")
            
            # Detect eyes directly using Haar cascades
            h, w = gray.shape
            eyes = self.eye_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(int(w/12), int(h/12)),
                maxSize=(int(w/3), int(h/3))
            )
            
            # If we don't have at least 2 eyes, try with different parameters
            if len(eyes) < 2:
                eyes = self.eye_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=5,
                    minSize=(int(w/15), int(h/15)),
                    maxSize=(int(w/2.5), int(h/2.5))
                )
                
            # If we still don't have enough eyes, try one more time with more sensitive parameters
            if len(eyes) < 2:
                eyes = self.eye_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(int(w/20), int(h/20)),
                    maxSize=(int(w/2), int(h/2))
                )
                
            # If we still don't have enough eyes, return empty measurements
            if len(eyes) < 2:
                if self.testing:
                    print("Not enough eyes detected in frame") 
                return frame, (None, None), (None, None)
                
            # Sort eyes by x-coordinate (left to right)
            eyes = sorted(eyes, key=lambda x: x[0])
            
            # Take only the first two eyes (assuming these are left and right)
            eyes = eyes[:2]
            
            # Create cropped eye images
            left_eye_rect = eyes[0]  # Format: (x, y, width, height)
            right_eye_rect = eyes[1]
            
            # Extract eye images
            left_eye_x, left_eye_y, left_eye_w, left_eye_h = left_eye_rect
            right_eye_x, right_eye_y, right_eye_w, right_eye_h = right_eye_rect
            
            left_eye_image = original_frame[left_eye_y:left_eye_y+left_eye_h, left_eye_x:left_eye_x+left_eye_w]
            right_eye_image = original_frame[right_eye_y:right_eye_y+right_eye_h, right_eye_x:right_eye_x+right_eye_w]
            
            # Fallback to approximate lid positions when facial landmarks aren't available
            left_upper_lid = (left_eye_x + left_eye_w//2, left_eye_y + int(left_eye_h * 0.25))
            left_lower_lid = (left_eye_x + left_eye_w//2, left_eye_y + int(left_eye_h * 0.75))
            
            right_upper_lid = (right_eye_x + right_eye_w//2, right_eye_y + int(right_eye_h * 0.25))
            right_lower_lid = (right_eye_x + right_eye_w//2, right_eye_y + int(right_eye_h * 0.75))
        
        # Add eye images to storage
        self.left_eye_frames.append(left_eye_image)
        self.right_eye_frames.append(right_eye_image)
        
        # Save the eye images
        if self.saving:
            left_eye_filename = os.path.join(self.output_dir, f"left_eye_{int(time.time())}.jpg")
            right_eye_filename = os.path.join(self.output_dir, f"right_eye_{int(time.time())}.jpg")
            cv2.imwrite(left_eye_filename, left_eye_image)
            cv2.imwrite(right_eye_filename, right_eye_image)
        
        # Use SAM for pupil detection if available, otherwise use image processing to find pupil
        if self.use_calibration and not for_calibration:
            # Try to detect iris (and thus pupil) using SAM
            left_pupil_center, right_pupil_center = self.detect_pupils_with_sam(left_eye_image, right_eye_image)
            
            # If SAM detection failed, use backup methods
            if left_pupil_center is None:
                left_pupil_center = self.detect_pupil_in_eye(left_eye_image)
                if left_pupil_center:
                    left_pupil_center = (left_eye_x + left_pupil_center[0], left_eye_y + left_pupil_center[1])
                else:
                    # Fallback to geometric center if detection fails
                    left_pupil_center = (left_eye_x + left_eye_w//2, left_eye_y + left_eye_h//2)
            
            if right_pupil_center is None:
                right_pupil_center = self.detect_pupil_in_eye(right_eye_image)
                if right_pupil_center:
                    right_pupil_center = (right_eye_x + right_pupil_center[0], right_eye_y + right_pupil_center[1])
                else:
                    # Fallback to geometric center if detection fails
                    right_pupil_center = (right_eye_x + right_eye_w//2, right_eye_y + right_eye_h//2)
        else:
            # Use conventional image processing for pupil detection
            left_pupil = self.detect_pupil_in_eye(left_eye_image)
            if left_pupil:
                left_pupil_center = (left_eye_x + left_pupil[0], left_eye_y + left_pupil[1])
            else:
                # Fallback to geometric center
                left_pupil_center = (left_eye_x + left_eye_w//2, left_eye_y + left_eye_h//2)
                
            right_pupil = self.detect_pupil_in_eye(right_eye_image)
            if right_pupil:
                right_pupil_center = (right_eye_x + right_pupil[0], right_eye_y + right_pupil[1])
            else:
                # Fallback to geometric center
                right_pupil_center = (right_eye_x + right_eye_w//2, right_eye_y + right_eye_h//2)
        
        # Calculate distances in pixels
        left_vph_px = self.calculate_distance(left_upper_lid, left_lower_lid)  # vertical palpebral height
        left_mrd1_px = self.calculate_distance(left_upper_lid, left_pupil_center)  # margin reflex distance 1
        
        right_vph_px = self.calculate_distance(right_upper_lid, right_lower_lid)  # vertical palpebral height
        right_mrd1_px = self.calculate_distance(right_upper_lid, right_pupil_center)  # margin reflex distance 1
        
        # Convert to millimeters if calibrated
        if self.is_calibrated and not for_calibration:
            left_vph = left_vph_px / self.left_pixels_per_mm
            left_mrd1 = left_mrd1_px / self.left_pixels_per_mm
            right_vph = right_vph_px / self.right_pixels_per_mm
            right_mrd1 = right_mrd1_px / self.right_pixels_per_mm
            
            # Add unit label for display
            left_vph_display = f"{left_vph:.2f} mm"
            left_mrd1_display = f"{left_mrd1:.2f} mm"
            right_vph_display = f"{right_vph:.2f} mm"
            right_mrd1_display = f"{right_mrd1:.2f} mm"
        else:
            # If not calibrated, just use pixel values
            left_vph = left_vph_px
            left_mrd1 = left_mrd1_px
            right_vph = right_vph_px
            right_mrd1 = right_mrd1_px
            
            # Add unit label for display
            left_vph_display = f"{left_vph:.2f} px"
            left_mrd1_display = f"{left_mrd1:.2f} px"
            right_vph_display = f"{right_vph:.2f} px"
            right_mrd1_display = f"{right_mrd1:.2f} px"
        
        # Draw measurements on frame to show video with measurements
        cv2.rectangle(frame, (left_eye_x, left_eye_y), (left_eye_x+left_eye_w, left_eye_y+left_eye_h), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_eye_x, right_eye_y), (right_eye_x+right_eye_w, right_eye_y+right_eye_h), (0, 255, 0), 2)
        
        cv2.line(frame, left_upper_lid, left_lower_lid, (0, 255, 0), 1)
        cv2.line(frame, left_pupil_center, left_upper_lid, (255, 0, 0), 1)
        cv2.line(frame, right_upper_lid, right_lower_lid, (0, 255, 0), 1)
        cv2.line(frame, right_pupil_center, right_upper_lid, (255, 0, 0), 1)
        
        cv2.circle(frame, left_pupil_center, 2, (255, 255, 0), -1)  # Yellow dot for pupil
        cv2.circle(frame, right_pupil_center, 2, (255, 255, 0), -1)  # Yellow dot for pupil
        cv2.circle(frame, left_upper_lid, 2, (0, 0, 255), -1)  # Red dot for upper lid
        cv2.circle(frame, left_lower_lid, 2, (0, 0, 255), -1)  # Red dot for lower lid
        cv2.circle(frame, right_upper_lid, 2, (0, 0, 255), -1)  # Red dot for upper lid
        cv2.circle(frame, right_lower_lid, 2, (0, 0, 255), -1)  # Red dot for lower lid
        
        # Store the raw measurements without unit labels
        measurements = ((left_vph, left_mrd1), (right_vph, right_mrd1))
        
        # Add calibration status indicator if calibrated
        calibration_status = "Calibrated (mm)" if self.is_calibrated else "Uncalibrated (px)"
        cv2.putText(frame, f"Status: {calibration_status}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame, measurements[0], measurements[1]
    
    def detect_pupils_with_sam(self, left_eye_image, right_eye_image):
        """Use SAM to detect pupils in eye images"""
        if not self.use_calibration:
            return None, None
            
        try:
            if self.sam_predictor is None:
                if not self.initialize_sam_model():
                    return None, None
                    
            # Process left eye
            left_pupil_center = None
            if left_eye_image is not None and left_eye_image.size > 0:
                # Convert BGR to RGB for SAM
                rgb_image = cv2.cvtColor(left_eye_image, cv2.COLOR_BGR2RGB)
                
                # Set the image for the predictor
                self.sam_predictor.set_image(rgb_image)
                
                # Generate a center point for the pupil (approximate center of the image)
                h, w = left_eye_image.shape[:2]
                center_point = np.array([[w//2, h//2]])
                
                # Get segmentation mask using the center point
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=center_point,
                    point_labels=np.array([1]),  # 1 for foreground
                    multimask_output=True
                )
                
                # Find the mask with the highest score
                best_mask_idx = np.argmax(scores)
                pupil_mask = masks[best_mask_idx]
                
                # Extract contours from the mask
                pupil_mask = pupil_mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # If contours found, calculate pupil center
                if contours:
                    # Find the largest contour (should be the pupil)
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Find the center of the contour
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        left_pupil_center = (cx, cy)
            
            # Process right eye
            right_pupil_center = None
            if right_eye_image is not None and right_eye_image.size > 0:
                # Similar process for right eye
                rgb_image = cv2.cvtColor(right_eye_image, cv2.COLOR_BGR2RGB)
                self.sam_predictor.set_image(rgb_image)
                
                h, w = right_eye_image.shape[:2]
                center_point = np.array([[w//2, h//2]])
                
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=center_point,
                    point_labels=np.array([1]),
                    multimask_output=True
                )
                
                best_mask_idx = np.argmax(scores)
                pupil_mask = masks[best_mask_idx]
                
                pupil_mask = pupil_mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        right_pupil_center = (cx, cy)
            
            return left_pupil_center, right_pupil_center
            
        except Exception as e:
            if self.testing:
                print(f"Error in SAM pupil detection: {e}")
            return None, None
    
    def detect_pupil_in_eye(self, eye_image):
        """Detect pupil using conventional image processing"""
        if eye_image is None or eye_image.size == 0:
            return None
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization to enhance contrast
            gray = cv2.equalizeHist(gray)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            
            # Use binary threshold to isolate dark regions (pupil)
            _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour (assumed to be the pupil)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate the center of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
            
            # If pupil detection fails, return None
            return None
            
        except Exception as e:
            if self.testing:
                print(f"Error in conventional pupil detection: {e}")
            return None
    
    def run(self, duration=20):
        """Run eye tracking for live video"""
        start_time = time.time()
        
        # Run calibration if enabled
        if self.use_calibration:
            print("Starting calibration...")
            calibration_success = self.calibrate()
            print(f"Calibration {'successful' if calibration_success else 'failed'}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_time = time.time() - start_time
            
            # Process frame
            frame, left_measurements, right_measurements = self.get_eye_measurements(frame)
            
            if left_measurements[0] is not None and right_measurements[0] is not None:
                # Store measurements
                self.measurements.append({
                    'timestamp': current_time,
                    'left_vph': left_measurements[0],
                    'left_mrd1': left_measurements[1],
                    'right_vph': right_measurements[0],
                    'right_mrd1': right_measurements[1],
                    'is_calibrated': self.is_calibrated
                })
                
                # Display measurements
                unit = "mm" if self.is_calibrated else "px"
                cv2.putText(frame, f"Left VPH (Vertical Palebral Height): {left_measurements[0]:.2f} {unit}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Left MRD1 (Margin to Reflex Distance 1): {left_measurements[1]:.2f} {unit}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right VPH (Vertical Palebral Height): {right_measurements[0]:.2f} {unit}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right MRD1 (Margin to Reflex Distance 1): {right_measurements[1]:.2f} {unit}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Eye Tracking - Live', frame)
            
            # Exit conditions
            if cv2.waitKey(1) & 0xFF == ord('q') or (current_time >= duration and self.file==0):
                break
        
        self.cleanup()
        csv_file = self.save_measurements()

        return csv_file

    def run_video(self):
        """Run eye tracking for uploaded video"""
        
        # Run calibration if enabled
        if self.use_calibration:
            print("Starting calibration...")
            calibration_success = self.calibrate()
            print(f"Calibration {'successful' if calibration_success else 'failed'}")
            
            # Rewind the video to the start
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            frame, left_measurements, right_measurements = self.get_eye_measurements(frame)
            
            if left_measurements[0] is not None and right_measurements[0] is not None:
                # Store measurements
                self.measurements.append({
                    'timestamp': self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,  # Convert to seconds
                    'left_palpebral_height': left_measurements[0],
                    'left_pupil_to_lower': left_measurements[1],
                    'right_palpebral_height': right_measurements[0],
                    'right_pupil_to_lower': right_measurements[1],
                    'is_calibrated': self.is_calibrated
                })
                
                # Display measurements
                unit = "mm" if self.is_calibrated else "px"
                cv2.putText(frame, f"Left Palpebral Height: {left_measurements[0]:.2f} {unit}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Left Pupil to Lower: {left_measurements[1]:.2f} {unit}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Palpebral Height: {right_measurements[0]:.2f} {unit}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Pupil to Lower: {right_measurements[1]:.2f} {unit}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Eye Tracking - Video', frame)
            
            # Exit conditions
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        csv_file = self.save_measurements()
        return csv_file

    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
    
    def save_measurements(self):
        """Save measurements to CSV file"""
        if self.measurements:
            df = pd.DataFrame(self.measurements)
            filename = os.path.join(self.output_dir, f"eye_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(filename, index=False)
            print(f"Measurements saved to {filename}")

            return filename

def main():
    
    print("Select an option:")
    print("1. Live Video")
    print("2. Upload Video")
    
    choice = input("Enter 1 or 2: ")
    
    if choice == '1':
        print("Starting eye tracking for live video...")
        print("Press 'q' to quit early")
        tracker = EyeTracker(0, saving=True, testing=False, use_calibration=True)
        tracker.run(duration=20)
    elif choice == '2':
        video_path = input("Enter the path to the video file: ")
        tracker = EyeTracker(video_path, saving=True, testing=False, use_calibration=True)
        tracker.run()
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()


## TODO:
# - fix the video upload eye capture