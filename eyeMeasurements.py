import cv2
import dlib
import numpy as np
import time
from datetime import datetime
import pandas as pd
import os

class EyeTracker:
    def __init__(self):
        # Initialize face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Storage for measurements
        self.measurements = []
        
        # Create a directory to save frames and measurements
        self.output_dir = "eye_tracking_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_eye_measurements(self, frame):
        """Extract eye measurements from a single frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        for face in faces:
            landmarks = self.predictor(gray, face)
            
            # Get left eye landmarks
            left_eye_points = []
            for n in range(36, 42):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                left_eye_points.append((x, y))
            
            # Get right eye landmarks
            right_eye_points = []
            for n in range(42, 48):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                right_eye_points.append((x, y))
            
            # Calculate measurements for left eye
            left_upper_lid = left_eye_points[1]  # Point 37
            left_lower_lid = left_eye_points[4]  # Point 40
            left_pupil_center = (
                (left_eye_points[0][0] + left_eye_points[3][0]) // 2,
                (left_eye_points[1][1] + left_eye_points[4][1]) // 2
            )
            
            # Calculate distances for left eye
            left_palpebral_height = self.calculate_distance(left_upper_lid, left_lower_lid)
            left_pupil_to_lower = self.calculate_distance(left_pupil_center, left_lower_lid)
            
            # Calculate measurements for right eye
            right_upper_lid = right_eye_points[1]  # Point 43
            right_lower_lid = right_eye_points[4]  # Point 46
            right_pupil_center = (
                (right_eye_points[0][0] + right_eye_points[3][0]) // 2,
                (right_eye_points[1][1] + right_eye_points[4][1]) // 2
            )
            
            # Calculate distances for right eye
            right_palpebral_height = self.calculate_distance(right_upper_lid, right_lower_lid)
            right_pupil_to_lower = self.calculate_distance(right_pupil_center, right_lower_lid)
            
            # Draw measurements on frame
            cv2.line(frame, left_upper_lid, left_lower_lid, (0, 255, 0), 1)
            cv2.line(frame, left_pupil_center, left_lower_lid, (255, 0, 0), 1)
            cv2.line(frame, right_upper_lid, right_lower_lid, (0, 255, 0), 1)
            cv2.line(frame, right_pupil_center, right_lower_lid, (255, 0, 0), 1)
            
            # Draw points for left eye
            for point in left_eye_points:
                cv2.circle(frame, point, 2, (0, 0, 255), -1)  # Red dot for left eye
            cv2.circle(frame, left_pupil_center, 2, (255, 255, 0), -1)  # Yellow dot for left pupil
            
            # Draw points for right eye
            for point in right_eye_points:
                cv2.circle(frame, point, 2, (0, 0, 255), -1)  # Red dot for right eye
            cv2.circle(frame, right_pupil_center, 2, (255, 255, 0), -1)  # Yellow dot for right pupil
            
            # Create larger bounding boxes around eyes
            left_eye_rect = cv2.boundingRect(np.array(left_eye_points))
            right_eye_rect = cv2.boundingRect(np.array(right_eye_points))
            
            # Increase the size of the bounding box
            left_eye_rect = (left_eye_rect[0] - 20, left_eye_rect[1] - 10, 
                             left_eye_rect[2] + 40, left_eye_rect[3] + 20)  # Adjust as needed
            right_eye_rect = (right_eye_rect[0] - 20, right_eye_rect[1] - 10, 
                              right_eye_rect[2] + 40, right_eye_rect[3] + 20)  # Adjust as needed
            
            # Save the eye regions as images
            left_eye_image = frame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3], 
                                    left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
            right_eye_image = frame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3], 
                                     right_eye_rect[0]:right_eye_rect[0] + right_eye_rect[2]]
            
            # Save the images to the output directory
            left_eye_filename = os.path.join(self.output_dir, f"left_eye_{int(time.time())}.jpg")
            right_eye_filename = os.path.join(self.output_dir, f"right_eye_{int(time.time())}.jpg")
            cv2.imwrite(left_eye_filename, left_eye_image)
            cv2.imwrite(right_eye_filename, right_eye_image)
            
            return frame, (left_palpebral_height, left_pupil_to_lower), (right_palpebral_height, right_pupil_to_lower)
        
        return frame, (None, None), (None, None)
    
    def run_live(self, duration=20):
        """Run eye tracking for live video"""
        start_time = time.time()
        
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
                cv2.putText(frame, f"Left Palpebral Height: {left_measurements[0]:.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Left Pupil to Lower: {left_measurements[1]:.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Palpebral Height: {right_measurements[0]:.2f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Pupil to Lower: {right_measurements[1]:.2f}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Eye Tracking - Live', frame)
            
            # Exit conditions
            if cv2.waitKey(1) & 0xFF == ord('q') or current_time >= duration:
                break
        
        self.cleanup()
        self.save_measurements()

    def run_video(self, video_path):
        """Run eye tracking for uploaded video"""
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame, left_measurements, right_measurements = self.get_eye_measurements(frame)
            
            if left_measurements[0] is not None and right_measurements[0] is not None:
                # Store measurements
                self.measurements.append({
                    'timestamp': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,  # Convert to seconds
                    'left_palpebral_height': left_measurements[0],
                    'left_pupil_to_lower': left_measurements[1],
                    'right_palpebral_height': right_measurements[0],
                    'right_pupil_to_lower': right_measurements[1]
                })
                
                # Display measurements
                cv2.putText(frame, f"Left Palpebral Height: {left_measurements[0]:.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Left Pupil to Lower: {left_measurements[1]:.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Palpebral Height: {right_measurements[0]:.2f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Pupil to Lower: {right_measurements[1]:.2f}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Eye Tracking - Video', frame)
            
            # Exit conditions
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        self.save_measurements()

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

def main():
    tracker = EyeTracker()
    
    print("Select an option:")
    print("1. Live Video")
    print("2. Upload Video")
    
    choice = input("Enter 1 or 2: ")
    
    if choice == '1':
        print("Starting eye tracking for live video...")
        print("Press 'q' to quit early")
        tracker.run_live(duration=20)
    elif choice == '2':
        video_path = input("Enter the path to the video file: ")
        tracker.run_video(video_path)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()


## TODO:
# - fix the video upload eye capture
