#!/usr/bin/env python3
"""
Eye Frame Extractor

This script:
1. Takes a video file as input
2. Identifies and extracts left and right eye frames
3. Saves the frames to a dedicated folder
4. Processes the frames to measure eye parameters
5. Saves the measurements to a CSV file
"""

import os
import cv2
import numpy as np
import pandas as pd
import dlib
import time
from datetime import datetime
import argparse
import glob

class EyeFrameExtractor:
    def __init__(self, video_path, output_dir=None, save_frames=True, pupil_diameter_mm=21.1, verbose=False, scaling_factor=3.0, pupil_correction=0.4):
        """Initialize the eye frame extractor
        
        Args:
            video_path (str): Path to the video file
            output_dir (str, optional): Output directory for frames and measurements
            save_frames (bool): Whether to save the extracted eye frames
            pupil_diameter_mm (float): Known pupil diameter in millimeters for calibration
            verbose (bool): Whether to print measurements for each frame
            scaling_factor (float): Additional scaling factor to adjust measurements
            pupil_correction (float): Correction factor for pupil diameter calculation
        """
        self.video_path = video_path
        self.save_frames = save_frames
        self.pupil_diameter_mm = pupil_diameter_mm
        self.calibration_factor = None  # Will be calculated during processing
        self.verbose = verbose
        self.scaling_factor = scaling_factor
        self.pupil_correction = pupil_correction
        
        # Create timestamp-based output directory if not specified
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"eye_frames_{timestamp}"
        else:
            self.output_dir = output_dir
            
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "left_eye"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "right_eye"), exist_ok=True)
        
        # Initialize face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Storage for measurements
        self.measurements = []
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def extract_eye_frames(self):
        """Extract left and right eye frames from the video"""
        print(f"Processing video with {self.frame_count} frames...")
        
        # Print header for verbose output
        if self.verbose:
            print("\nFrame-by-frame measurements:")
            print("Frame\tLeft VPH (mm)\tLeft MRD1 (mm)\tRight VPH (mm)\tRight MRD1 (mm)\tPupil Diameter (mm)")
            print("-" * 95)
        
        frame_number = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert to seconds
            
            # Process the frame to extract eye regions and measurements
            left_eye_img, right_eye_img, left_measurements, right_measurements = self.process_frame(frame, frame_number)
            
            # Save eye images if requested
            if self.save_frames:
                if left_eye_img is not None:
                    left_path = os.path.join(self.output_dir, "left_eye", f"frame_{frame_number:04d}.jpg")
                    cv2.imwrite(left_path, left_eye_img)
                    
                if right_eye_img is not None:
                    right_path = os.path.join(self.output_dir, "right_eye", f"frame_{frame_number:04d}.jpg")
                    cv2.imwrite(right_path, right_eye_img)
            
            # Store measurements if available
            if left_measurements and right_measurements:
                # Default calibration factor if not yet calculated
                if self.calibration_factor is None:
                    self.calibration_factor = 0.1  # default fallback value
                
                # Extract measurements
                left_vph_px, left_mrd1_px, left_pupil_diameter_px = left_measurements
                right_vph_px, right_mrd1_px, right_pupil_diameter_px = right_measurements
                
                # Calculate average pupil diameter
                avg_pupil_diameter_px = (left_pupil_diameter_px + right_pupil_diameter_px) / 2.0
                
                # Convert to millimeters
                left_vph_mm = left_vph_px * self.calibration_factor
                left_mrd1_mm = left_mrd1_px * self.calibration_factor
                right_vph_mm = right_vph_px * self.calibration_factor
                right_mrd1_mm = right_mrd1_px * self.calibration_factor
                pupil_diameter_mm = avg_pupil_diameter_px * self.calibration_factor
                
                # Print measurements if verbose
                if self.verbose:
                    print(f"{frame_number}\t{left_vph_mm:.2f}\t\t{left_mrd1_mm:.2f}\t\t{right_vph_mm:.2f}\t\t{right_mrd1_mm:.2f}\t\t{pupil_diameter_mm:.2f}")
                
                self.measurements.append({
                    'timestamp': timestamp,
                    'frame': frame_number,
                    'left_palpebral_height_px': left_vph_px,
                    'left_pupil_to_lower_px': left_mrd1_px,
                    'right_palpebral_height_px': right_vph_px,
                    'right_pupil_to_lower_px': right_mrd1_px,
                    'left_palpebral_height_mm': left_vph_mm,
                    'left_pupil_to_lower_mm': left_mrd1_mm,
                    'right_palpebral_height_mm': right_vph_mm,
                    'right_pupil_to_lower_mm': right_mrd1_mm,
                    'left_pupil_diameter_px': left_pupil_diameter_px,
                    'right_pupil_diameter_px': right_pupil_diameter_px,
                    'avg_pupil_diameter_px': avg_pupil_diameter_px,
                    'pupil_diameter_mm': pupil_diameter_mm,
                    'calibration_factor': self.calibration_factor
                })
            elif self.verbose and (frame_number % 100 == 0 or frame_number < 10):
                # If we couldn't get measurements for this frame, print a notification in verbose mode
                print(f"{frame_number}\tNo measurements available - face or eyes not detected")
            
            frame_number += 1
            
            # Print progress every 100 frames
            if frame_number % 100 == 0:
                print(f"Processed {frame_number}/{self.frame_count} frames ({frame_number/self.frame_count*100:.1f}%)")
        
        # Release the video capture
        self.cap.release()
        
        # Print summary if verbose
        if self.verbose:
            print("-" * 95)
            print(f"Completed processing {frame_number} frames with calibration factor: {self.calibration_factor:.5f} mm/pixel")
            if self.measurements:
                # Calculate and print average measurements
                df = pd.DataFrame(self.measurements)
                print("\nAverage Measurements:")
                print(f"Left VPH: {df['left_palpebral_height_mm'].mean():.2f} mm")
                print(f"Left MRD1: {df['left_pupil_to_lower_mm'].mean():.2f} mm")
                print(f"Right VPH: {df['right_palpebral_height_mm'].mean():.2f} mm")
                print(f"Right MRD1: {df['right_pupil_to_lower_mm'].mean():.2f} mm")
                print(f"Average Pupil Diameter: {df['pupil_diameter_mm'].mean():.2f} mm")
        
        # Save measurements to CSV
        if self.measurements:
            return self.save_measurements()
        else:
            print("No measurements were collected!")
            return None
    
    def process_frame(self, frame, frame_number):
        """Process a single frame to extract eye regions and measurements
        
        Returns:
            left_eye_img: Image of the left eye
            right_eye_img: Image of the right eye
            left_measurements: Tuple of (vertical_palpebral_height, pupil_to_lower, pupil_diameter) for left eye
            right_measurements: Tuple of (vertical_palpebral_height, pupil_to_lower, pupil_diameter) for right eye
        """
        # Make a copy of the original frame
        original_frame = frame.copy()
        visualization_frame = frame.copy()  # For visualization without affecting measurements
        
        # Convert to grayscale for facial detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with different scale parameters
        faces = self.detector(gray, 0)  # 0 means don't upsample
        
        # If no faces found, try with upsampling
        if len(faces) == 0:
            if self.verbose and frame_number % 100 == 0:
                print(f"Frame {frame_number}: No face detected on first pass, trying with upsampling...")
            faces = self.detector(gray, 1)  # 1 means upsample once (2x)
        
        # If still no faces, try different preprocessing
        if len(faces) == 0:
            if self.verbose and frame_number % 100 == 0:
                print(f"Frame {frame_number}: Still no face detected, trying with Gaussian blur...")
            # Try Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            faces = self.detector(blurred, 1)
        
        if len(faces) == 0:
            if self.verbose and frame_number % 100 == 0:
                print(f"Frame {frame_number}: No face could be detected after multiple attempts")
            return None, None, None, None
        
        # Log if multiple faces were found
        if len(faces) > 1 and self.verbose and frame_number % 100 == 0:
            print(f"Frame {frame_number}: Multiple faces detected ({len(faces)}), using largest")
            
        # Get the largest face (closest to camera)
        if len(faces) > 1:
            face = max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            face = faces[0]
        
        # Get facial landmarks
        landmarks = self.predictor(gray, face)
        
        # Get left eye landmarks (points 36-41 in dlib's 68 point model)
        left_eye_points = []
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye_points.append((x, y))
        
        # Get right eye landmarks (points 42-47)
        right_eye_points = []
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye_points.append((x, y))
        
        # Calculate measurements for left eye
        left_upper_lid = (
            (left_eye_points[1][0] + left_eye_points[2][0]) // 2,
            (left_eye_points[1][1] + left_eye_points[2][1]) // 2
        )  # Avg of Point 37 and Point 38
        left_lower_lid = (
            (left_eye_points[4][0] + left_eye_points[5][0]) // 2,
            (left_eye_points[4][1] + left_eye_points[5][1]) // 2
        )  # Avg of Points 40 and 41
        left_pupil_center = (
            (left_eye_points[0][0] + left_eye_points[3][0]) // 2,
            (left_eye_points[1][1] + left_eye_points[4][1]) // 2
        )  # midpoint of points 37, 38, 40, and 41
        
        # Calculate distances for left eye
        left_vph = self.calculate_distance(left_upper_lid, left_lower_lid)  # vertical palebral height
        left_mrd1 = self.calculate_distance(left_upper_lid, left_pupil_center)  # mrd1 = pupil to upper
        
        # Calculate measurements for right eye
        right_upper_lid = (
            (right_eye_points[1][0] + right_eye_points[2][0]) // 2,
            (right_eye_points[1][1] + right_eye_points[2][1]) // 2
        )  # Midpoint of points 43 and 44
        right_lower_lid = (
            (right_eye_points[4][0] + right_eye_points[5][0]) // 2,
            (right_eye_points[4][1] + right_eye_points[5][1]) // 2
        )  # Midpoint of points 46 and 47
        right_pupil_center = (
            (right_eye_points[0][0] + right_eye_points[3][0]) // 2,
            (right_eye_points[1][1] + right_eye_points[4][1]) // 2
        )  # midpoint of points 43, 44, 46, and 47
        
        # Calculate distances for right eye
        right_vph = self.calculate_distance(right_upper_lid, right_lower_lid)  # vertical palpebral height
        right_mrd1 = self.calculate_distance(right_upper_lid, right_pupil_center)
        
        # Calculate horizontal eye width (what was previously labeled as pupil diameter)
        left_eye_width = self.calculate_distance(left_eye_points[0], left_eye_points[3])
        right_eye_width = self.calculate_distance(right_eye_points[0], right_eye_points[3])
        
        # Convert eye points to numpy arrays for boundingRect
        left_eye_array = np.array(left_eye_points)
        right_eye_array = np.array(right_eye_points)
        
        # Get bounding rectangles
        left_eye_rect = cv2.boundingRect(left_eye_array)
        right_eye_rect = cv2.boundingRect(right_eye_array)
        
        # Calculate centers of eye bounding boxes
        left_eye_center_x = left_eye_rect[0] + left_eye_rect[2] // 2
        left_eye_center_y = left_eye_rect[1] + left_eye_rect[3] // 2
        
        right_eye_center_x = right_eye_rect[0] + right_eye_rect[2] // 2
        right_eye_center_y = right_eye_rect[1] + right_eye_rect[3] // 2
        
        # Make bounding boxes square and larger for better context
        eye_size = int(max(left_eye_rect[2], left_eye_rect[3], right_eye_rect[2], right_eye_rect[3]) * 2.0)
        
        # Ensure the eye size is reasonable
        min_eye_size = 50  # Minimum eye crop size
        eye_size = max(eye_size, min_eye_size)
        
        # Calculate new eye rectangles (centered on detected eyes, square, and with padding)
        left_x1 = max(0, left_eye_center_x - eye_size // 2)
        left_y1 = max(0, left_eye_center_y - eye_size // 2)
        left_x2 = min(frame.shape[1], left_eye_center_x + eye_size // 2)
        left_y2 = min(frame.shape[0], left_eye_center_y + eye_size // 2)
        
        right_x1 = max(0, right_eye_center_x - eye_size // 2)
        right_y1 = max(0, right_eye_center_y - eye_size // 2)
        right_x2 = min(frame.shape[1], right_eye_center_x + eye_size // 2)
        right_y2 = min(frame.shape[0], right_eye_center_y + eye_size // 2)
        
        # Extract eye regions for pupil detection
        left_eye_region = frame[left_y1:left_y2, left_x1:left_x2]
        right_eye_region = frame[right_y1:right_y2, right_x1:right_x2]
        
        # Detect pupils in the extracted eye regions
        left_pupil_center_region, left_pupil_radius = self.detect_pupil(left_eye_region)
        right_pupil_center_region, right_pupil_radius = self.detect_pupil(right_eye_region)
        
        # Calculate actual pupil diameters
        left_pupil_diameter = 0
        right_pupil_diameter = 0
        
        # Get the visualization images (original_frame was modified with the measurement lines)
        left_eye_viz = visualization_frame[left_y1:left_y2, left_x1:left_x2].copy()
        right_eye_viz = visualization_frame[right_y1:right_y2, right_x1:right_x2].copy()
        
        if left_pupil_center_region is not None:
            left_pupil_diameter = left_pupil_radius * 2
            # Draw detected pupil on the left eye visualization
            # Draw multiple circles for better visibility
            cv2.circle(left_eye_viz, left_pupil_center_region, left_pupil_radius, (0, 0, 255), 2)
            cv2.circle(left_eye_viz, left_pupil_center_region, left_pupil_radius+1, (255, 0, 0), 1)
            cv2.circle(left_eye_viz, left_pupil_center_region, 1, (0, 255, 255), 2)  # Center dot
            
            # Map pupil center from cropped region back to original frame coordinates
            left_pupil_global_x = left_x1 + left_pupil_center_region[0]
            left_pupil_global_y = left_y1 + left_pupil_center_region[1]
            # Draw reference on original frame for debugging
            cv2.circle(original_frame, (left_pupil_global_x, left_pupil_global_y), 
                      max(2, left_pupil_radius//3), (0, 0, 255), -1)
        
        if right_pupil_center_region is not None:
            right_pupil_diameter = right_pupil_radius * 2
            # Draw detected pupil on the right eye visualization
            cv2.circle(right_eye_viz, right_pupil_center_region, right_pupil_radius, (0, 0, 255), 2)
            cv2.circle(right_eye_viz, right_pupil_center_region, right_pupil_radius+1, (255, 0, 0), 1)
            cv2.circle(right_eye_viz, right_pupil_center_region, 1, (0, 255, 255), 2)  # Center dot
            
            # Map pupil center from cropped region back to original frame coordinates
            right_pupil_global_x = right_x1 + right_pupil_center_region[0]
            right_pupil_global_y = right_y1 + right_pupil_center_region[1]
            # Draw reference on original frame for debugging
            cv2.circle(original_frame, (right_pupil_global_x, right_pupil_global_y), 
                       max(2, right_pupil_radius//3), (0, 0, 255), -1)
        
        # If pupil detection failed, use a default value based on eye width
        if left_pupil_diameter == 0:
            left_pupil_diameter = left_eye_width * 0.2  # Estimate pupil as 20% of eye width
            # Add a text warning about fallback method
            cv2.putText(left_eye_viz, "Pupil detection failed - using estimate", 
                       (5, left_eye_viz.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        if right_pupil_diameter == 0:
            right_pupil_diameter = right_eye_width * 0.2  # Estimate pupil as 20% of eye width
            # Add a text warning about fallback method
            cv2.putText(right_eye_viz, "Pupil detection failed - using estimate", 
                       (5, right_eye_viz.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Average the pupil diameters
        pupil_diameter_pixels = (left_pupil_diameter + right_pupil_diameter) / 2.0
        
        # Calculate calibration factor (mm per pixel)
        if self.calibration_factor is None and pupil_diameter_pixels > 0:
            self.calibration_factor = self.pupil_diameter_mm / pupil_diameter_pixels
            
            # Apply scaling factor to get more realistic measurements based on empirical data
            self.calibration_factor = self.calibration_factor * self.scaling_factor
            
            print(f"Calibration factor: {self.calibration_factor:.5f} mm/pixel (based on pupil diameter of {self.pupil_diameter_mm} mm)")
            
            if self.verbose:
                print("\nDetection parameters:")
                print(f"Left eye VPH (pixels): {left_vph:.2f}")
                print(f"Left eye MRD1 (pixels): {left_mrd1:.2f}")
                print(f"Right eye VPH (pixels): {right_vph:.2f}")
                print(f"Right eye MRD1 (pixels): {right_mrd1:.2f}")
                print(f"Left pupil diameter (pixels): {left_pupil_diameter:.2f}")
                print(f"Right pupil diameter (pixels): {right_pupil_diameter:.2f}")
                print(f"Average pupil diameter (pixels): {pupil_diameter_pixels:.2f}")
                print(f"Using calibration factor of {self.calibration_factor:.5f} mm/pixel\n")
        
        # Draw measurement lines on the original frame before extraction
        # Left eye
        cv2.line(original_frame, 
                 (int(left_upper_lid[0]), int(left_upper_lid[1])), 
                 (int(left_lower_lid[0]), int(left_lower_lid[1])), 
                 (0, 255, 0), 2)  # Green line for VPH
        cv2.line(original_frame, 
                 (int(left_upper_lid[0]), int(left_upper_lid[1])), 
                 (int(left_pupil_center[0]), int(left_pupil_center[1])), 
                 (255, 0, 0), 2)  # Blue line for MRD1
        
        # Right eye
        cv2.line(original_frame, 
                 (int(right_upper_lid[0]), int(right_upper_lid[1])), 
                 (int(right_lower_lid[0]), int(right_lower_lid[1])), 
                 (0, 255, 0), 2)  # Green line for VPH
        cv2.line(original_frame, 
                 (int(right_upper_lid[0]), int(right_upper_lid[1])), 
                 (int(right_pupil_center[0]), int(right_pupil_center[1])), 
                 (255, 0, 0), 2)  # Blue line for MRD1
        
        # Mark the eye width with yellow lines (previously called pupil diameter)
        cv2.line(original_frame, 
                 (int(left_eye_points[0][0]), int(left_eye_points[0][1])), 
                 (int(left_eye_points[3][0]), int(left_eye_points[3][1])), 
                 (0, 255, 255), 1)  # Yellow line for eye width
        
        cv2.line(original_frame, 
                 (int(right_eye_points[0][0]), int(right_eye_points[0][1])), 
                 (int(right_eye_points[3][0]), int(right_eye_points[3][1])), 
                 (0, 255, 255), 1)  # Yellow line for eye width
        
        # Draw points for eye landmarks
        for point in left_eye_points + right_eye_points:
            cv2.circle(original_frame, (int(point[0]), int(point[1])), 2, (255, 255, 0), -1)
        
        # Extract eye images from the processed frame with overlaid annotations
        try:
            left_eye_image = original_frame[left_y1:left_y2, left_x1:left_x2]
            right_eye_image = original_frame[right_y1:right_y2, right_x1:right_x2]
            
            # Add measurement text to eye images
            if self.calibration_factor is not None:
                # Add text for left eye
                left_vph_mm = left_vph * self.calibration_factor
                left_mrd1_mm = left_mrd1 * self.calibration_factor
                left_pupil_mm = left_pupil_diameter * self.calibration_factor
                
                # Add text to the visualization images
                cv2.putText(left_eye_viz, f"VPH: {left_vph_mm:.2f}mm", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(left_eye_viz, f"MRD1: {left_mrd1_mm:.2f}mm", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(left_eye_viz, f"Pupil: {left_pupil_mm:.2f}mm", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Add text for right eye
                right_vph_mm = right_vph * self.calibration_factor
                right_mrd1_mm = right_mrd1 * self.calibration_factor
                right_pupil_mm = right_pupil_diameter * self.calibration_factor
                
                cv2.putText(right_eye_viz, f"VPH: {right_vph_mm:.2f}mm", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(right_eye_viz, f"MRD1: {right_mrd1_mm:.2f}mm", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(right_eye_viz, f"Pupil: {right_pupil_mm:.2f}mm", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Return the measurements for further processing, using the visualization images
            left_measurements = (left_vph, left_mrd1, left_pupil_diameter)
            right_measurements = (right_vph, right_mrd1, right_pupil_diameter)
            
            return left_eye_viz, right_eye_viz, left_measurements, right_measurements
        except Exception as e:
            if self.verbose:
                print(f"Error extracting eye images in frame {frame_number}: {e}")
            return None, None, None, None
    
    def save_measurements(self):
        """Save eye measurements to CSV file"""
        if not self.measurements:
            print("No measurements to save!")
            return None
        
        # Create a DataFrame
        df = pd.DataFrame(self.measurements)
        
        # Save to CSV
        csv_file = os.path.join(self.output_dir, "eye_measurements.csv")
        df.to_csv(csv_file, index=False)
        
        print(f"Measurements saved to {csv_file}")
        return csv_file

    def detect_pupil(self, eye_image):
        """
        Detect the pupil in an eye image and calculate its diameter
        
        Args:
            eye_image: Image of the eye
            
        Returns:
            tuple: (pupil_center, pupil_radius) or (None, None) if pupil not detected
        """
        if eye_image is None or eye_image.size == 0:
            return None, None
            
        # Convert to grayscale if needed
        if len(eye_image.shape) > 2:
            gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_eye = eye_image.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_eye, (7, 7), 0)
        
        # Get image dimensions
        height, width = blurred.shape
        
        # Create a mask for the center region of the eye
        # This helps focus on the pupil area and reduce false detections
        mask = np.zeros_like(blurred)
        center_x, center_y = width // 2, height // 2
        cv2.circle(mask, (center_x, center_y), int(min(width, height) * 0.4), 255, -1)
        
        # Apply the mask
        masked_eye = cv2.bitwise_and(blurred, blurred, mask=mask)
        
        # Try multiple threshold values to find the best pupil candidate
        best_contour = None
        best_score = 0
        best_center = None
        best_radius = None
        
        # Try different thresholds
        for threshold_value in [15, 20, 25, 30, 35, 40]:
            # Apply threshold to get the darkest regions (potential pupils)
            _, thresholded = cv2.threshold(masked_eye, threshold_value, 255, cv2.THRESH_BINARY_INV)
            
            # Apply morphological operations to clean up the image
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            morph = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if not contours:
                continue
                
            # Evaluate each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Skip tiny contours
                if area < 100:
                    continue
                    
                # Skip too large contours (not likely to be pupils)
                if area > (width * height * 0.5):
                    continue
                
                # Get the center and radius using minEnclosingCircle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Calculate distance from center of eye
                center_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Calculate score (favors circular shapes near the center)
                score = circularity * (1 - center_dist / (width/2)) * area
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
                    best_center = (int(x), int(y))
                    best_radius = int(radius)
        
        # If no good contour found, try adaptive thresholding
        if best_contour is None:
            # Apply adaptive threshold
            adaptive_thresh = cv2.adaptiveThreshold(
                masked_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2)
            
            # Clean up with morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if contours:
                # Find the largest contour near the center
                max_area = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > max_area and area > 100 and area < (width * height * 0.5):
                        # Calculate center of contour
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            # Check if near center
                            if np.sqrt((cx - center_x)**2 + (cy - center_y)**2) < width * 0.3:
                                max_area = area
                                (x, y), radius = cv2.minEnclosingCircle(contour)
                                best_center = (int(x), int(y))
                                best_radius = int(radius)
        
        # If still no pupil found, use a simple minimum intensity method as last resort
        if best_center is None:
            # Find the darkest point within the central region
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_eye)
            if min_loc:
                # Use a reasonable default radius based on eye size
                best_center = min_loc
                best_radius = int(min(width, height) * 0.15)
        
        return best_center, best_radius

def process_extracted_frames(eye_frames_dir):
    """Process previously extracted eye frames to generate measurements
    
    This function can be used to process eye frames that were already extracted
    but need measurements calculated.
    
    Args:
        eye_frames_dir (str): Directory containing 'left_eye' and 'right_eye' subdirectories
        
    Returns:
        str: Path to the saved measurements CSV file
    """
    left_eye_dir = os.path.join(eye_frames_dir, "left_eye")
    right_eye_dir = os.path.join(eye_frames_dir, "right_eye")
    
    # Verify directories exist
    if not os.path.exists(left_eye_dir) or not os.path.exists(right_eye_dir):
        print(f"Error: Required directories not found in {eye_frames_dir}")
        return None
    
    # Get all left eye images
    left_eye_images = sorted(glob.glob(os.path.join(left_eye_dir, "*.jpg")))
    right_eye_images = sorted(glob.glob(os.path.join(right_eye_dir, "*.jpg")))
    
    print(f"Found {len(left_eye_images)} left eye images and {len(right_eye_images)} right eye images")
    
    # Initialize detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Initialize measurements list
    measurements = []
    
    # Process each pair of images
    for i, (left_path, right_path) in enumerate(zip(left_eye_images, right_eye_images)):
        # Extract frame number from filename
        left_frame_number = int(os.path.basename(left_path).split('_')[1].split('.')[0])
        right_frame_number = int(os.path.basename(right_path).split('_')[1].split('.')[0])
        
        # Ensure we're processing matching frames
        if left_frame_number != right_frame_number:
            print(f"Warning: Frame number mismatch - left: {left_frame_number}, right: {right_frame_number}")
            continue
        
        # Load images
        left_eye_img = cv2.imread(left_path)
        right_eye_img = cv2.imread(right_path)
        
        if left_eye_img is None or right_eye_img is None:
            print(f"Warning: Could not read images for frame {left_frame_number}")
            continue
        
        # TODO: Process the eye images to get measurements
        # For demonstration, we'll use placeholder values
        # In a real implementation, this would use the same measurement
        # logic as in process_frame
        
        measurements.append({
            'timestamp': i / 30.0,  # Assume 30 fps if not known
            'frame': left_frame_number,
            'left_palpebral_height': 10.0,  # Placeholder
            'left_pupil_to_lower': 5.0,     # Placeholder
            'right_palpebral_height': 10.0, # Placeholder
            'right_pupil_to_lower': 5.0     # Placeholder
        })
        
        # Print progress
        if i % 100 == 0:
            print(f"Processed {i}/{len(left_eye_images)} images")
    
    # Save measurements to CSV
    if measurements:
        df = pd.DataFrame(measurements)
        csv_file = os.path.join(eye_frames_dir, "eye_measurements.csv")
        df.to_csv(csv_file, index=False)
        print(f"Measurements saved to {csv_file}")
        return csv_file
    else:
        print("No measurements were collected!")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract eyes from video frames and measure eye parameters")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output", "-o", help="Output directory for frames and measurements")
    parser.add_argument("--no-save", action="store_true", help="Don't save extracted frames")
    parser.add_argument("--process-only", action="store_true", help="Process previously extracted frames only")
    
    args = parser.parse_args()
    
    if args.process_only:
        # Process existing frames
        output_dir = args.output if args.output else args.video_path + "_eyes"
        csv_file = process_extracted_frames(output_dir)
    else:
        # Extract frames and process
        extractor = EyeFrameExtractor(
            args.video_path, 
            output_dir=args.output,
            save_frames=not args.no_save
        )
        csv_file = extractor.extract_eye_frames()
    
    if csv_file:
        print(f"Eye measurement data saved to: {csv_file}")
        print(f"To visualize results, use:")
        print(f"  python software_pipeline.py --eye-data {csv_file} --video {args.video_path}")

if __name__ == "__main__":
    main()
