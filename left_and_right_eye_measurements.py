import cv2
import dlib
import numpy as np
import time
from datetime import datetime
import pandas as pd
import os

class EyeTracker:
    def __init__(self, file=0, saving=False, testing=False):
        # Initialize face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(file)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Storage for measurements
        self.measurements = []
        self.left_eye_frames = list()
        self.right_eye_frames = list()
        
        # Create a directory to save frames and measurements with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"eye_tracking_output_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        # For debugging
        if testing:
            print(f"Created output directory: {self.output_dir}")

        # save constructor inputs
        self.testing = testing
        self.saving = saving
        self.file = file
        self.frame_count = 0
        
    def get_output_dir(self):
        """return the output directory containing the images"""
        return self.output_dir
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_eye_measurements(self, frame):
        """Extract eye measurements from a single frame"""
        if frame is None or frame.size == 0:
            if self.testing:
                print("Empty frame received!")
            return frame, (None, None), (None, None)
            
        # Make a copy of the original frame for saving
        original_frame = frame.copy()
        
        # Convert to grayscale for facial detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply some preprocessing to improve detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with different scale parameters for better detection
        faces = self.detector(gray, 0)  # 0 means don't upsample
        
        # If no faces found, try with upsampling
        if len(faces) == 0:
            faces = self.detector(gray, 1)  # 1 means upsample once (2x)
        
        # If still no faces, try different preprocessing
        if len(faces) == 0:
            # Try Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            faces = self.detector(blurred, 1)
        
        if len(faces) == 0:
            if self.testing:
                print(f"No faces detected in frame {self.frame_count}")
            return frame, (None, None), (None, None)
        
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
        right_mrd1 = self.calculate_distance(right_upper_lid, right_pupil_center)  # mrd1 = upper lid to pupil
        
        # Convert eye points to numpy arrays for boundingRect
        left_eye_array = np.array(left_eye_points)
        right_eye_array = np.array(right_eye_points)
        
        # Get basic bounding rectangles
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
        
        # Extract eye images
        try:
            left_eye_image = original_frame[left_y1:left_y2, left_x1:left_x2]
            right_eye_image = original_frame[right_y1:right_y2, right_x1:right_x2]
            
            # Check if the eye images are valid
            if left_eye_image.size == 0 or right_eye_image.size == 0:
                if self.testing:
                    print(f"Empty eye image in frame {self.frame_count}")
                return frame, (left_vph, left_mrd1), (right_vph, right_mrd1)
                
            # Append to the list of eye frames
            self.left_eye_frames.append(left_eye_image)
            self.right_eye_frames.append(right_eye_image)
            
            # Save the eye images
            if self.saving:
                timestamp = int(time.time() * 1000)  # Millisecond timestamp
                frame_id = f"{self.frame_count:04d}"
                
                left_eye_path = os.path.join(self.output_dir, f"left_eye_{frame_id}_{timestamp}.jpg")
                right_eye_path = os.path.join(self.output_dir, f"right_eye_{frame_id}_{timestamp}.jpg")
                
                # Check if the image is valid
                if left_eye_image.size > 0 and left_eye_image.shape[0] > 0 and left_eye_image.shape[1] > 0:
                    cv2.imwrite(left_eye_path, left_eye_image)
                if right_eye_image.size > 0 and right_eye_image.shape[0] > 0 and right_eye_image.shape[1] > 0:
                    cv2.imwrite(right_eye_path, right_eye_image)
                
                if self.testing:
                    print(f"Saved eye images for frame {self.frame_count}: {left_eye_path}, {right_eye_path}")
                    print(f"Left eye shape: {left_eye_image.shape}, Right eye shape: {right_eye_image.shape}")
        except Exception as e:
            if self.testing:
                print(f"Error extracting eye images: {e}")
        
        # Draw bounding boxes on the frame for visualization
        cv2.rectangle(frame, (left_x1, left_y1), (left_x2, left_y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (right_x1, right_y1), (right_x2, right_y2), (0, 255, 0), 2)
        
        # Draw measurements on frame
        cv2.line(frame, left_upper_lid, left_lower_lid, (0, 255, 0), 1)
        cv2.line(frame, left_pupil_center, left_upper_lid, (255, 0, 0), 1)
        cv2.line(frame, right_upper_lid, right_lower_lid, (0, 255, 0), 1)
        cv2.line(frame, right_pupil_center, right_upper_lid, (255, 0, 0), 1)
        
        # Draw points for left eye
        for point in left_eye_points:
            cv2.circle(frame, point, 2, (0, 0, 255), -1)  # Red dot for left eye
        cv2.circle(frame, left_pupil_center, 2, (255, 255, 0), -1)  # Yellow dot for left pupil
        
        # Draw points for right eye
        for point in right_eye_points:
            cv2.circle(frame, point, 2, (0, 0, 255), -1)  # Red dot for right eye
        cv2.circle(frame, right_pupil_center, 2, (255, 255, 0), -1)  # Yellow dot for right pupil
        
        # Increment frame counter
        self.frame_count += 1
        
        return frame, (left_vph, left_mrd1), (right_vph, right_mrd1)
    
    def run(self, duration=20):
        """Run eye tracking for live video or video file"""
        start_time = time.time()
        
        if isinstance(self.file, int):  # Live camera
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
                        'left_palpebral_height': left_measurements[0],
                        'left_pupil_to_lower': left_measurements[1],
                        'right_palpebral_height': right_measurements[0],
                        'right_pupil_to_lower': right_measurements[1]
                    })
                    
                    # Display measurements
                    cv2.putText(frame, f"Left VPH: {left_measurements[0]:.2f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Left MRD1: {left_measurements[1]:.2f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Right VPH: {right_measurements[0]:.2f}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Right MRD1: {right_measurements[1]:.2f}", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Eye Tracking', frame)
                
                # Exit conditions
                if cv2.waitKey(1) & 0xFF == ord('q') or (current_time >= duration):
                    break
        else:  # Video file
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
                        'right_pupil_to_lower': right_measurements[1]
                    })
                    
                    # Display measurements
                    cv2.putText(frame, f"Left VPH: {left_measurements[0]:.2f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Left MRD1: {left_measurements[1]:.2f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Right VPH: {right_measurements[0]:.2f}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Right MRD1: {right_measurements[1]:.2f}", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Eye Tracking', frame)
                
                # Exit conditions
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        self.cleanup()
        csv_file = self.save_measurements()
        
        return csv_file
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
    
    def save_measurements(self):
        """Save eye measurements to CSV file"""
        if not self.measurements:
            print("No measurements to save!")
            return None
        
        # Create a DataFrame
        df = pd.DataFrame(self.measurements)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(self.output_dir, f"eye_measurements_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        
        print(f"Measurements saved to {csv_file}")
        return csv_file

def main():
    tracker = EyeTracker(0, saving=True, testing=True)  # Use camera
    tracker.run()

if __name__ == "__main__":
    main()


## TODO:
# - fix the video upload eye capture