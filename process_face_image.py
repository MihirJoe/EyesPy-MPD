import cv2
import numpy as np
import os
import dlib

def process_face_image():
    """Process a face image to get accurate eye measurements"""
    # Create output directory
    output_dir = "eye_tracking_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to save a test image
    image_path = os.path.join(output_dir, "face_image.jpg")
    
    # Create a test image from the video if available
    if os.path.exists("avneesh_blinking.mov"):
        cap = cv2.VideoCapture("avneesh_blinking.mov")
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            cv2.imwrite(image_path, frame)
            print(f"Extracted frame from video: {image_path}")
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Initialize facial landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Detect faces
    faces = detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    print(f"Detected {len(faces)} faces")
    
    if len(faces) == 0:
        print("No faces detected")
        return
    
    # Process the first face
    face = faces[0]
    
    # Get facial landmarks
    landmarks = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), face)
    
    # Left eye points (36-41 in the 68-point model)
    left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    
    # Right eye points (42-47 in the 68-point model)
    right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
    
    # Calculate upper and lower lid points for left eye
    left_upper_lid = ((landmarks.part(37).x + landmarks.part(38).x) // 2, 
                     (landmarks.part(37).y + landmarks.part(38).y) // 2)
    left_lower_lid = ((landmarks.part(40).x + landmarks.part(41).x) // 2, 
                     (landmarks.part(40).y + landmarks.part(41).y) // 2)
    
    # Calculate pupil center for left eye (average of all eye points)
    left_pupil_x = sum(p[0] for p in left_eye_points) // 6
    left_pupil_y = sum(p[1] for p in left_eye_points) // 6
    left_pupil_center = (left_pupil_x, left_pupil_y)
    
    # Calculate upper and lower lid points for right eye
    right_upper_lid = ((landmarks.part(43).x + landmarks.part(44).x) // 2, 
                      (landmarks.part(43).y + landmarks.part(44).y) // 2)
    right_lower_lid = ((landmarks.part(46).x + landmarks.part(47).x) // 2, 
                      (landmarks.part(46).y + landmarks.part(47).y) // 2)
    
    # Calculate pupil center for right eye (average of all eye points)
    right_pupil_x = sum(p[0] for p in right_eye_points) // 6
    right_pupil_y = sum(p[1] for p in right_eye_points) // 6
    right_pupil_center = (right_pupil_x, right_pupil_y)
    
    # Calculate VPF (Vertical Palpebral Fissure) - distance between upper and lower lid
    left_vpf = np.sqrt((left_upper_lid[0] - left_lower_lid[0])**2 + 
                      (left_upper_lid[1] - left_lower_lid[1])**2)
    
    right_vpf = np.sqrt((right_upper_lid[0] - right_lower_lid[0])**2 + 
                       (right_upper_lid[1] - right_lower_lid[1])**2)
    
    # Calculate MRD1 (Margin Reflex Distance 1) - distance from upper lid to pupil center
    left_mrd1 = np.sqrt((left_upper_lid[0] - left_pupil_center[0])**2 + 
                       (left_upper_lid[1] - left_pupil_center[1])**2)
    
    right_mrd1 = np.sqrt((right_upper_lid[0] - right_pupil_center[0])**2 + 
                        (right_upper_lid[1] - right_pupil_center[1])**2)
    
    # Draw measurements on the image
    result_img = img.copy()
    
    # Draw eye landmarks
    for i in range(36, 42):
        cv2.circle(result_img, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)
    
    for i in range(42, 48):
        cv2.circle(result_img, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 0), -1)
    
    # Draw upper and lower eyelids
    cv2.circle(result_img, left_upper_lid, 3, (0, 0, 255), -1)
    cv2.circle(result_img, left_lower_lid, 3, (0, 0, 255), -1)
    cv2.circle(result_img, right_upper_lid, 3, (0, 0, 255), -1)
    cv2.circle(result_img, right_lower_lid, 3, (0, 0, 255), -1)
    
    # Draw pupil centers
    cv2.circle(result_img, left_pupil_center, 3, (255, 0, 0), -1)
    cv2.circle(result_img, right_pupil_center, 3, (255, 0, 0), -1)
    
    # Draw VPF lines
    cv2.line(result_img, left_upper_lid, left_lower_lid, (0, 255, 0), 2)
    cv2.line(result_img, right_upper_lid, right_lower_lid, (0, 255, 0), 2)
    
    # Draw MRD1 lines
    cv2.line(result_img, left_upper_lid, left_pupil_center, (255, 0, 0), 2)
    cv2.line(result_img, right_upper_lid, right_pupil_center, (255, 0, 0), 2)
    
    # Add text with measurements
    cv2.putText(result_img, f"Left VPF: {left_vpf:.2f} px", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(result_img, f"Left MRD1: {left_mrd1:.2f} px", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(result_img, f"Right VPF: {right_vpf:.2f} px", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(result_img, f"Right MRD1: {right_mrd1:.2f} px", (10, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Print measurements
    print(f"Left VPF: {left_vpf:.2f} px")
    print(f"Left MRD1: {left_mrd1:.2f} px")
    print(f"Right VPF: {right_vpf:.2f} px")
    print(f"Right MRD1: {right_mrd1:.2f} px")
    
    # Save the result
    result_path = os.path.join(output_dir, "accurate_measurements.jpg")
    cv2.imwrite(result_path, result_img)
    print(f"Saved result to: {result_path}")
    
    # Display the result
    cv2.imshow("Eye Measurements", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_face_image() 