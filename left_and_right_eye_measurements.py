import cv2
import dlib
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from datetime import datetime
import pandas as pd
import os

testing = False # True to print helpful messages when debugging

class EyeTracker:
    def __init__(self, saving=False, testing = False):
        # Initialize face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Create a GUI app 
        self.app = tk.Tk() 
        self.app.title("Video Upload")
        
        # Bind the app with Escape keyboard to 
        # quit app whenever pressed 
        self.app.bind('<Escape>', lambda e: self.app.quit()) 
        
        # Create a label and display it on app 
        self.label_widget = tk.Label(self.app) 
        self.label_widget.pack() 
        
        # Storage for measurements
        self.measurements = []
        self.left_eye_frames = list()
        self.right_eye_frames = list()
        
        # Create a directory to save frames and measurements
        self.output_dir = "eye_tracking_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # save constructor inputs
        self.testing= testing
        self.saving = saving

        
    def get_output_dir(self):
        """return the output directory containing the images"""
        return self.output_dir
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_eye_measurements(self, frame):
        """Extract eye measurements from a single frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        originalFrame = frame
        
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
            left_upper_lid = (
                (left_eye_points[1][0] + left_eye_points[2][0]) // 2,
                (left_eye_points[1][1] + left_eye_points[2][1]) // 2
            )# Avg of Point 37 and Point 38
            left_lower_lid = (
                (left_eye_points[4][0] + left_eye_points[5][0]) // 2,
                (left_eye_points[4][1] + left_eye_points[5][1]) // 2
            ) # Avg of Points 40 and 41
            left_pupil_center = (
                (left_eye_points[0][0] + left_eye_points[3][0]) // 2,
                (left_eye_points[1][1] + left_eye_points[4][1]) // 2
            ) # midpoint of points 37, 38, 40, and 41
            
            # Calculate distances for left eye
            left_vph = self.calculate_distance(left_upper_lid, left_lower_lid) #vertical palebral height
            left_mrd1 = self.calculate_distance(left_upper_lid, left_pupil_center) #mrd1 = pupil to upper
            
            # Calculate measurements for right eye
            right_upper_lid = (
                (right_eye_points[1][0] + right_eye_points[2][0]) // 2,
                (right_eye_points[1][1] + right_eye_points[2][1]) // 2
            ) # Midpoint of points 43 and 44
            right_lower_lid = (
                (right_eye_points[4][0] + right_eye_points[5][0]) // 2,
                (right_eye_points[4][1] + right_eye_points[5][1]) // 2
            ) # Midpoint of points 46 and 47
            right_pupil_center = (
                (right_eye_points[0][0] + right_eye_points[3][0]) // 2,
                (right_eye_points[1][1] + right_eye_points[4][1]) // 2
            ) # midpoint of points 43, 44, 46, and 47
            
            # Calculate distances for right eye
            right_vph = self.calculate_distance(right_upper_lid, right_lower_lid) # vertical palpebral height
            right_mrd1 = self.calculate_distance(right_upper_lid, right_pupil_center) # mrd1 = upper lid to pupil
            
            # Create larger bounding boxes around eyes
            left_eye_rect = cv2.boundingRect(np.array(left_eye_points))
            right_eye_rect = cv2.boundingRect(np.array(right_eye_points))
            
            # Increase the size of the bounding box and make square
            if self.testing:
                print(f'Left Eye Rectange: {left_eye_rect}')
            left_eye_width = int(1.5 * max(left_eye_rect[2:3]))
            left_eye_offset = int(left_eye_width/2)

            right_eye_width = int(1.5 * max(right_eye_rect[2:3]))
            right_eye_offset = int(right_eye_width/2)

            left_eye_rect = (int(left_eye_rect[0] + left_eye_rect[2]/2 - left_eye_offset), int(left_eye_rect[1] + left_eye_rect[3]/2 - left_eye_offset), 
                             left_eye_width, left_eye_width)  # Adjust as needed
            right_eye_rect = (int(right_eye_rect[0] + right_eye_rect[2]/2 - right_eye_offset), int(right_eye_rect[1] + right_eye_rect[3]/2 - right_eye_offset), 
                             right_eye_width, right_eye_width)  # Adjust as needed
            
            # Save the eye regions as images
            left_eye_image = frame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3], 
                                    left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
            right_eye_image = frame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3], 
                                     right_eye_rect[0]:right_eye_rect[0] + right_eye_rect[2]]
            self.left_eye_frames.append(left_eye_image)
            self.right_eye_frames.append(right_eye_image)
            
            # Save the images to the output directory
            if self.saving:
                left_eye_image = originalFrame[left_eye_rect[1]:left_eye_rect[1] + left_eye_rect[3], 
                                    left_eye_rect[0]:left_eye_rect[0] + left_eye_rect[2]]
                right_eye_image = originalFrame[right_eye_rect[1]:right_eye_rect[1] + right_eye_rect[3], 
                                     right_eye_rect[0]:right_eye_rect[0] + right_eye_rect[2]]
                left_eye_filename = os.path.join(self.output_dir, f"left_eye_{int(time.time())}.jpg")
                right_eye_filename = os.path.join(self.output_dir, f"right_eye_{int(time.time())}.jpg")
                cv2.imwrite(left_eye_filename, left_eye_image)
                cv2.imwrite(right_eye_filename, right_eye_image)

            # Draw measurements on frame to show video with measurements
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
            
            return frame, (left_vph, left_mrd1), (right_vph, right_mrd1)
        
        return frame, (None, None), (None, None)
    
    def run_video(self, duration=20):
        """Run eye tracking for the video in self.cap"""
        
        ret, frame = self.cap.read()
        if not ret:
            self.csv_file = self.save_measurements()
            self.cleanup()
            return
        
        current_time = time.time() - self.start_time
        
        # Process frame
        frame, left_measurements, right_measurements = self.get_eye_measurements(frame)
        
        if left_measurements[0] is not None and right_measurements[0] is not None:
            # Store measurements
            self.measurements.append({
                'timestamp': current_time,
                'left_vph': left_measurements[0],
                'left_mrd1': left_measurements[1],
                'right_vph': right_measurements[0],
                'right_mrd1': right_measurements[1]
            })
            
            # Display measurements
            cv2.putText(frame, f"Left VPH (Vertical Palebral Height): {left_measurements[0]:.2f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Left MRD1 (Margin to Reflex Distance 1): {left_measurements[1]:.2f}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Right VPH (Vertical Palebral Height): {right_measurements[0]:.2f}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Right MRD1 (Margin to Reflex Distance 1): {right_measurements[1]:.2f}", 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert image from one color space to other 
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) 
    
        # Capture the latest frame and transform to image 
        captured_image = Image.fromarray(opencv_image) 
    
        # Convert captured image to photoimage 
        photo_image = ImageTk.PhotoImage(image=captured_image) 
    
        # Displaying photoimage in the label 
        self.label_widget.photo_image = photo_image 
    
        # Configure image in the label 
        self.label_widget.configure(image=photo_image) 

        # Exit conditions
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (current_time >= duration and self.filename==0):
            self.csv_file = self.save_measurements()
            self.cleanup()
            return
    
        # Repeat the same process after every 10 seconds if exit conditions are not met
        self.label_widget.after(10, self.run_video) 

        """ Have to rewrite to work in tkinter using .after method... Sucks. But will work with GUI once done. 
            https://www.geeksforgeeks.org/how-to-show-webcam-in-tkinter-window-python/ """
        # cv2.imshow('Eye Tracking - Live', frame)
            


    def run_live(self):
        """Set up cv2.cap object for live video collection"""
        
        # no file, filename = 0
        self.filename = 0

        # Initialize video capture and setup tkinter window
        self.cap = cv2.VideoCapture(self.filename)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Declare the width and height in variables 
        width, height = 800, 600
        
        # Set the width and height 
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width) 
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 


        self.start_time = time.time()
        self.run_video()

    
    def get_file(self):
        """Allow tkinter to let user choose a file."""
        # Define allowable filetypes
        filetypes = (
            ('MOV files', '*.mov'),
            ('MP4 files', '*.mp4'),
            ('Other Video files', '*.*')
        )

        self.filename = filedialog.askopenfilename(
            title='Open a Video or Image File',
            initialdir='./',
            filetypes=filetypes)

        # Initialize video capture and setup tkinter window
        self.cap = cv2.VideoCapture(self.filename)
        
        # Declare the width and height in variables 
        width, height = 800, 600
        
        # Set the width and height 
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width) 
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 

        # run the video through the algorithm
        self.start_time = time.time()
        self.run_video()


    def cleanup(self):
        """Release resources"""
        self.app.quit()
        self.cap.release()

    
    def save_measurements(self):
        """Save measurements to CSV file"""
        if self.measurements:
            df = pd.DataFrame(self.measurements)
            filename = os.path.join(self.output_dir, f"eye_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(filename, index=False)
            print(f"Measurements saved to {filename}")

            return filename
        
    def run(self):
        """Create interactive GUI and use it to start data collection from the uploaded video or from the live video"""
        # Create a button to open the camera in GUI app 
        button1 = tk.Button(self.app, text="Take Live Video", command=self.run_live) 
        button1.pack() 

        button2 = tk.Button(self.app, text="Upload Video File", command=self.get_file)
        button2.pack()
        
        # Create an infinite loop for displaying app on screen 
        self.app.mainloop() 

def main():
    tracker = EyeTracker(saving=True, testing = False)
    tracker.run()


if __name__ == "__main__":
    main()
    if testing:
        test = input("Is the video still open?")


## TODO:
# - fix the video to show the measurements
# - ensure the live video capture still works
# - why does it not close right after running? 
