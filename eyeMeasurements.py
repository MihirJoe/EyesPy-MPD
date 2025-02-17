import cv2
import dlib
import numpy as np
import time
from datetime import datetime
import pandas as pd

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
            
            # Calculate measurements
            upper_lid = left_eye_points[1]  # Point 37
            lower_lid = left_eye_points[4]  # Point 40
            pupil_center = (
                (left_eye_points[0][0] + left_eye_points[3][0]) // 2,
                (left_eye_points[1][1] + left_eye_points[4][1]) // 2
            )
            
            # Calculate distances
            palpebral_height = self.calculate_distance(upper_lid, lower_lid)
            pupil_to_lower = self.calculate_distance(pupil_center, lower_lid)
            
            # Draw measurements on frame
            cv2.line(frame, upper_lid, lower_lid, (0, 255, 0), 1)
            cv2.line(frame, pupil_center, lower_lid, (255, 0, 0), 1)
            
            # Draw points
            for point in left_eye_points:
                cv2.circle(frame, point, 2, (0, 0, 255), -1)
            cv2.circle(frame, pupil_center, 2, (255, 255, 0), -1)
            
            return frame, palpebral_height, pupil_to_lower
        
        return frame, None, None
    
    def run(self, duration=20):
        """Run eye tracking for specified duration"""
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_time = time.time() - start_time
            
            # Process frame
            frame, palpebral_height, pupil_to_lower = self.get_eye_measurements(frame)
            
            if palpebral_height and pupil_to_lower:
                # Store measurements
                self.measurements.append({
                    'timestamp': current_time,
                    'palpebral_height': palpebral_height,
                    'pupil_to_lower': pupil_to_lower
                })
                
                # Display measurements
                cv2.putText(frame, f"Palpebral Height: {palpebral_height:.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Pupil to Lower: {pupil_to_lower:.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {current_time:.1f}s", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Eye Tracking', frame)
            
            # Exit conditions
            if cv2.waitKey(1) & 0xFF == ord('q') or current_time >= duration:
                break
        
        self.cleanup()
        self.save_measurements()
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
    
    def save_measurements(self):
        """Save measurements to CSV file"""
        if self.measurements:
            df = pd.DataFrame(self.measurements)
            filename = f"eye_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"Measurements saved to {filename}")

if __name__ == "__main__":
    # Create and run tracker
    tracker = EyeTracker()
    print("Starting eye tracking for 20 seconds...")
    print("Press 'q' to quit early")
    tracker.run(duration=20)
