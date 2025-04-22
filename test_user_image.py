import cv2
import numpy as np
import os
from left_and_right_eye_measurements import EyeTracker

def save_user_image():
    """Create and save an image similar to what the user provided"""
    # Ensure output directory exists
    os.makedirs("eye_tracking_output", exist_ok=True)
    
    # Path to save the image
    image_path = "eye_tracking_output/test_eye_image.png"
    
    # Read the video file if it exists (we noticed you have a video file that might contain similar faces)
    video_path = "avneesh_blinking.mov"
    if os.path.exists(video_path):
        try:
            # Extract a frame from the video
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                cv2.imwrite(image_path, frame)
                print(f"Saved frame from video to: {image_path}")
                return image_path
        except Exception as e:
            print(f"Error extracting frame from video: {e}")
    
    # If we couldn't use the video, create a face image
    img_height = 800
    img_width = 600
    face_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    face_img[:] = (220, 210, 200)  # Light skin tone background
    
    # Draw face outline
    face_center_x = img_width // 2
    face_center_y = img_height // 2
    cv2.ellipse(face_img, (face_center_x, face_center_y), (180, 250), 0, 0, 360, (200, 180, 170), -1)
    
    # Draw hair
    hair_points = np.array([
        [100, 50], [500, 50], [550, 200], [50, 200]
    ], np.int32)
    hair_points = hair_points.reshape((-1, 1, 2))
    cv2.fillPoly(face_img, [hair_points], (40, 40, 40))
    
    # Draw eyebrows
    left_eyebrow_y = 300
    right_eyebrow_y = 300
    cv2.line(face_img, (face_center_x - 100, left_eyebrow_y), (face_center_x - 20, left_eyebrow_y - 10), (50, 50, 50), 5)
    cv2.line(face_img, (face_center_x + 20, right_eyebrow_y - 10), (face_center_x + 100, right_eyebrow_y), (50, 50, 50), 5)
    
    # Draw eyes
    left_eye_center_x = face_center_x - 70
    right_eye_center_x = face_center_x + 70
    eye_center_y = 350
    
    # Draw eye whites
    cv2.ellipse(face_img, (left_eye_center_x, eye_center_y), (30, 15), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(face_img, (right_eye_center_x, eye_center_y), (30, 15), 0, 0, 360, (255, 255, 255), -1)
    
    # Draw iris and pupil
    cv2.circle(face_img, (left_eye_center_x, eye_center_y), 12, (80, 60, 30), -1) # Brown iris
    cv2.circle(face_img, (left_eye_center_x, eye_center_y), 5, (0, 0, 0), -1) # Pupil
    cv2.circle(face_img, (right_eye_center_x, eye_center_y), 12, (80, 60, 30), -1) # Brown iris
    cv2.circle(face_img, (right_eye_center_x, eye_center_y), 5, (0, 0, 0), -1) # Pupil
    
    # Add highlights to eyes
    cv2.circle(face_img, (left_eye_center_x + 3, eye_center_y - 3), 2, (255, 255, 255), -1)
    cv2.circle(face_img, (right_eye_center_x + 3, eye_center_y - 3), 2, (255, 255, 255), -1)
    
    # Draw eyelids
    for i in range(5):
        cv2.ellipse(face_img, (left_eye_center_x, eye_center_y), (30 + i, 20 + i), 0, 0, 180, (200, 180, 170), 1)
        cv2.ellipse(face_img, (left_eye_center_x, eye_center_y), (30 + i, 18 + i), 0, 180, 360, (200, 180, 170), 1)
        cv2.ellipse(face_img, (right_eye_center_x, eye_center_y), (30 + i, 20 + i), 0, 0, 180, (200, 180, 170), 1)
        cv2.ellipse(face_img, (right_eye_center_x, eye_center_y), (30 + i, 18 + i), 0, 180, 360, (200, 180, 170), 1)
    
    # Draw nose
    nose_start_y = eye_center_y + 30
    nose_end_y = eye_center_y + 120
    cv2.line(face_img, (face_center_x, nose_start_y), (face_center_x, nose_end_y), (180, 160, 150), 2)
    cv2.ellipse(face_img, (face_center_x - 15, nose_end_y), (15, 10), 0, 0, 180, (180, 160, 150), 2)
    cv2.ellipse(face_img, (face_center_x + 15, nose_end_y), (15, 10), 0, 0, 180, (180, 160, 150), 2)
    
    # Draw mouth
    mouth_y = eye_center_y + 180
    cv2.ellipse(face_img, (face_center_x, mouth_y), (60, 20), 0, 0, 180, (150, 90, 90), 3)
    
    # Save the image
    cv2.imwrite(image_path, face_img)
    print(f"Created and saved test face image to: {image_path}")
    
    return image_path

def test_user_image():
    """Test eye measurement on the user-provided image"""
    # Create/save an image for testing
    image_path = save_user_image()
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    print(f"Image size: {img.shape}")
    
    # Create a tracker with testing enabled (for more verbose output)
    tracker = EyeTracker(file=0, saving=True, testing=True, use_calibration=False)
    
    # Process the frame - this will use our improved eye measurements with facial landmarks
    frame, left_measurements, right_measurements = tracker.get_eye_measurements(img)
    
    # Show results
    if left_measurements[0] is not None or right_measurements[0] is not None:
        print("Measurements found:")
        if left_measurements[0] is not None:
            print(f"Left VPH: {left_measurements[0]:.2f} px")
            print(f"Left MRD1: {left_measurements[1]:.2f} px")
        if right_measurements[0] is not None:
            print(f"Right VPH: {right_measurements[0]:.2f} px")
            print(f"Right MRD1: {right_measurements[1]:.2f} px")
    else:
        print("No measurements found")
    
    # Display the image with measurements
    cv2.imshow("Eye Measurements", frame)
    print("Displaying image with measurements. Press any key to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save output image
    output_dir = os.path.join("eye_tracking_output", "test_results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed_user_eye_image.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Processed image saved to: {output_path}")

if __name__ == "__main__":
    test_user_image() 