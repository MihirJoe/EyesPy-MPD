def find_iris_dark_blob(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest)
    return int(x), int(y), int(radius)

def detect_iris_center_and_radius(image):
    iris_circle = find_iris_with_hough(image)
    if iris_circle is not None:
        return iris_circle
    return find_iris_dark_blob(image)

def is_mask_circular(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter == 0:
        return False

    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return 0.7 < circularity < 1.3
import os
import cv2
import dlib
import numpy as np
import torch
import glob
from segment_anything import sam_model_registry, SamPredictor

def find_iris_with_hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=gray.shape[0] // 4,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=gray.shape[0] // 2
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        sorted_circles = sorted(circles[0, :], key=lambda c: c[2], reverse=True)
        best_circle = sorted_circles[0]
        center_x, center_y, radius = best_circle
        return center_x, center_y, radius
    else:
        return None

def find_dark_region_center(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(blurred)
    return minLoc  # Returns (x, y)

class IrisSegmentor:
    def __init__(self, use_sam=True):
        """
        Initialize the iris segmentation module.
        
        Args:
            use_sam (bool): Whether to use SAM for iris segmentation
        """
        # Initialize face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # SAM model setup
        self.use_sam = use_sam
        self.sam_predictor = None
        if self.use_sam:
            self.initialize_sam_model()
            
        # Create output directory
        self.output_dir = "mihir_test"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def initialize_sam_model(self):
        """Initialize the SAM model for iris segmentation"""
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
                print("SAM model weights not found. Please ensure a .pth file with 'sam' in the name exists.")
                self.use_sam = False
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
            self.use_sam = False
            return False
    
    def detect_eyes(self, image):
        """
        Detect eyes using facial landmarks.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing eye data or None if no faces detected
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        
        if len(faces) == 0:
            print("No faces detected in the image")
            return None
        
        # Get the first face
        face = faces[0]
        
        # Get facial landmarks
        landmarks = self.predictor(gray, face)
        
        # Extract eye landmarks
        # Left eye: points 36-41
        # Right eye: points 42-47
        left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        
        # Calculate eye regions with padding
        padding = 10
        
        # Left eye bounding box
        left_eye_x = max(0, min(p[0] for p in left_eye_points) - padding)
        left_eye_y = max(0, min(p[1] for p in left_eye_points) - padding)
        left_eye_w = max(p[0] for p in left_eye_points) - left_eye_x + padding
        left_eye_h = max(p[1] for p in left_eye_points) - left_eye_y + padding
        
        # Right eye bounding box
        right_eye_x = max(0, min(p[0] for p in right_eye_points) - padding)
        right_eye_y = max(0, min(p[1] for p in right_eye_points) - padding)
        right_eye_w = max(p[0] for p in right_eye_points) - right_eye_x + padding
        right_eye_h = max(p[1] for p in right_eye_points) - right_eye_y + padding
        
        # Ensure we don't go beyond image boundaries
        h, w = image.shape[:2]
        left_eye_w = min(left_eye_w, w - left_eye_x)
        left_eye_h = min(left_eye_h, h - left_eye_y)
        right_eye_w = min(right_eye_w, w - right_eye_x)
        right_eye_h = min(right_eye_h, h - right_eye_y)
        
        # Extract eye images
        left_eye_image = image[left_eye_y:left_eye_y+left_eye_h, left_eye_x:left_eye_x+left_eye_w]
        right_eye_image = image[right_eye_y:right_eye_y+right_eye_h, right_eye_x:right_eye_x+right_eye_w]
        
        # Store eye coordinates for reference
        left_eye_coords = (left_eye_x, left_eye_y, left_eye_w, left_eye_h)
        right_eye_coords = (right_eye_x, right_eye_y, right_eye_w, right_eye_h)
        
        return {
            'left_eye': {
                'image': left_eye_image,
                'coords': left_eye_coords
            },
            'right_eye': {
                'image': right_eye_image,
                'coords': right_eye_coords
            },
            'landmarks': landmarks
        }
    
    def split_eye_image(self, image):
        """
        Split an image containing both eyes into separate left and right eye images.
        Used for close-up images of eyes when face detection fails.
        
        Args:
            image: Input image containing both eyes
            
        Returns:
            Dictionary containing separate left and right eye data
        """
        h, w = image.shape[:2]
        midpoint = w // 2
        
        # Split image into left and right halves
        # Note: From the camera's perspective, the left half contains the right eye,
        # and the right half contains the left eye
        right_eye_image = image[:, :midpoint]
        left_eye_image = image[:, midpoint:]
        
        # Store eye coordinates
        right_eye_coords = (0, 0, midpoint, h)
        left_eye_coords = (midpoint, 0, w - midpoint, h)
        
        print(f"Split image into left eye ({left_eye_image.shape[:2]}) and right eye ({right_eye_image.shape[:2]})")
        
        # Save the split images for debugging
        cv2.imwrite(os.path.join(self.output_dir, "split_left_eye.png"), left_eye_image)
        cv2.imwrite(os.path.join(self.output_dir, "split_right_eye.png"), right_eye_image)
        
        return {
            'left_eye': {
                'image': left_eye_image,
                'coords': left_eye_coords
            },
            'right_eye': {
                'image': right_eye_image,
                'coords': right_eye_coords
            }
        }
    
    def process_direct_eye_image(self, image):
        """
        Process an image that already contains just the eye.
        Used when face detection fails or for close-up eye images.
        
        Args:
            image: Input image containing an eye
            
        Returns:
            Dictionary containing eye data
        """
        h, w = image.shape[:2]
        coords = (0, 0, w, h)  # Using the entire image
        
        return {
            'direct_eye': {
                'image': image,
                'coords': coords
            }
        }
    
    def segment_iris_with_sam(self, eye_image, eye_side="left"):
        if not self.use_sam or eye_image is None or eye_image.size == 0:
            return None, None

        try:
            h, w = eye_image.shape[:2]

            iris_circle = detect_iris_center_and_radius(eye_image)

            if iris_circle is None:
                print("No iris detected")
                return None, None

            center_x, center_y, radius = iris_circle

            pad = int(radius * 1.2)
            x1 = max(center_x - pad, 0)
            y1 = max(center_y - pad, 0)
            x2 = min(center_x + pad, w)
            y2 = min(center_y + pad, h)

            bbox = np.array([x1, y1, x2, y2])
            center_point = np.array([[center_x, center_y]])

            rgb_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(rgb_image)

            masks, scores, _ = self.sam_predictor.predict(
                point_coords=center_point,
                point_labels=np.array([1]),
                box=bbox[np.newaxis, :],
                multimask_output=True
            )

            best_mask_idx = np.argmax(scores)
            iris_mask = masks[best_mask_idx]

            iris_mask_uint8 = (iris_mask * 255).astype(np.uint8)
            
            if not is_mask_circular(iris_mask_uint8):
                print("Mask rejected: not circular enough")
                return None, None

            contours, _ = cv2.findContours(iris_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return iris_mask_uint8, None

            largest_contour = max(contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
            diameter = radius * 2

            return iris_mask_uint8, diameter

        except Exception as e:
            print(f"Error in SAM iris segmentation: {e}")
            return None, None
    
    def process_image(self, image_path, visualize=True):
        """
        Process an image to detect eyes and segment the iris.
        
        Args:
            image_path: Path to the input image
            visualize: Whether to create visualization images
            
        Returns:
            results: Dictionary containing results for both eyes
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        
        # Try to detect eyes in a face
        eye_data = self.detect_eyes(image)
        
        # If no face detected, try to split the image (assuming it contains both eyes)
        if eye_data is None:
            print("No face detected. Trying to split the image into left and right eyes...")
            eye_data = self.split_eye_image(image)
            eye_keys = ['left_eye', 'right_eye']
        else:
            eye_keys = ['left_eye', 'right_eye']
        
        results = {}
        for key in eye_keys:
            results[key] = {}
        
        # Process each eye
        for eye_key in eye_keys:
            eye_image = eye_data[eye_key]['image']
            eye_coords = eye_data[eye_key]['coords']

            # Segment iris using SAM, passing eye_side
            iris_mask, iris_diameter = self.segment_iris_with_sam(eye_image, eye_side=eye_key)

            if iris_mask is not None:
                # Save the mask
                mask_filename = os.path.join(self.output_dir, f"{eye_key}_iris_mask.png")
                cv2.imwrite(mask_filename, iris_mask)

                # Store results
                results[eye_key]['mask_path'] = mask_filename
                results[eye_key]['diameter'] = iris_diameter

                if visualize:
                    # Create visualization image
                    vis_image = eye_image.copy()

                    # Ensure mask and image have the same dimensions
                    if iris_mask.shape[:2] != eye_image.shape[:2]:
                        print(f"Warning: Mask shape {iris_mask.shape[:2]} doesn't match image shape {eye_image.shape[:2]}")
                        # Resize mask to match image dimensions if needed
                        iris_mask = cv2.resize(iris_mask, (eye_image.shape[1], eye_image.shape[0]))

                    # Create colored mask for visualization
                    color_mask = np.zeros_like(eye_image)
                    color_mask[iris_mask > 0] = [0, 255, 0]  # Green mask

                    # Blend original image and mask
                    alpha = 0.5
                    blended = cv2.addWeighted(vis_image, 1, color_mask, alpha, 0)

                    # If we have a diameter, draw the circle
                    if iris_diameter is not None:
                        # Find center of the iris mask
                        M = cv2.moments(iris_mask)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            radius = int(iris_diameter / 2)

                            # Draw circle representing the iris diameter
                            cv2.circle(blended, (cx, cy), radius, (255, 0, 0), 2)

                            # Add diameter text
                            cv2.putText(blended, f"Diameter: {iris_diameter:.2f} px", 
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # Save visualization
                    vis_filename = os.path.join(self.output_dir, f"{eye_key}_visualization.png")
                    cv2.imwrite(vis_filename, blended)
                    results[eye_key]['visualization_path'] = vis_filename
        
        return results

def main():
    # Create the iris segmentor
    segmentor = IrisSegmentor()
    
    # Get input image path
    image_path = input("Enter the path to the input image: ")
    
    # Process the image
    results = segmentor.process_image(image_path)
    
    if results:
        print("Processing completed successfully.")
        print("Results saved to directory:", segmentor.output_dir)
        
        # Display results
        for eye_key, eye_data in results.items():
            if 'diameter' in eye_data:
                print(f"{eye_key.replace('_', ' ').title()} iris diameter: {eye_data['diameter']:.2f} pixels")
    else:
        print("Error processing the image.")

if __name__ == "__main__":
    main()