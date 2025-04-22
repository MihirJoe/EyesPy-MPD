import cv2
import numpy as np
import os
from left_and_right_eye_measurements import EyeTracker

def save_test_image():
    """Save the test image from user input"""
    # Ensure output directory exists
    os.makedirs("eye_tracking_output", exist_ok=True)
    
    # Path to save the image
    image_path = "eye_tracking_output/face_image.jpg"
    
    # In a real scenario, this would save the image from the user input
    # For now, we'll extract a frame from the video if available
    if os.path.exists("avneesh_blinking.mov"):
        try:
            cap = cv2.VideoCapture("avneesh_blinking.mov")
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                cv2.imwrite(image_path, frame)
                print(f"Saved frame from video to: {image_path}")
                return image_path
        except Exception as e:
            print(f"Error extracting frame from video: {e}")
    
    # Otherwise create a test face image
    print("Creating a test face image...")
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
    
    # Draw eyebrows, nose, mouth
    cv2.line(face_img, (left_eye_center_x - 30, eye_center_y - 30), (left_eye_center_x + 30, eye_center_y - 30), (50, 50, 50), 5)
    cv2.line(face_img, (right_eye_center_x - 30, eye_center_y - 30), (right_eye_center_x + 30, eye_center_y - 30), (50, 50, 50), 5)
    cv2.line(face_img, (face_center_x, eye_center_y + 30), (face_center_x, eye_center_y + 100), (150, 130, 120), 3)
    cv2.ellipse(face_img, (face_center_x, eye_center_y + 180), (60, 20), 0, 0, 180, (150, 100, 100), 3)
    
    # Save the image
    cv2.imwrite(image_path, face_img)
    print(f"Created and saved test face image to: {image_path}")
    
    return image_path

def test_face_detection():
    """Test the improved eye detection on a face image"""
    # Save test image
    image_path = save_test_image()
    
    # Create eye tracker with testing enabled
    tracker = EyeTracker(file=0, saving=True, testing=True, use_calibration=True)
    
    # Read the test image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    print(f"Image size: {img.shape}")
    
    # First try to calibrate to get accurate measurements
    if tracker.use_calibration:
        print("Starting calibration...")
        calibration_success = tracker.calibrate()
        print(f"Calibration {'successful' if calibration_success else 'failed'}")
    
    # Process the image using our improved eye detection
    frame, left_measurements, right_measurements = tracker.get_eye_measurements(img)
    
    # Display the results
    if left_measurements[0] is not None or right_measurements[0] is not None:
        print("\nMeasurements found:")
        if left_measurements[0] is not None:
            unit = "mm" if tracker.is_calibrated else "px"
            print(f"Left VPF (Vertical Palpebral Height): {left_measurements[0]:.2f} {unit}")
            print(f"Left MRD1 (Margin Reflex Distance 1): {left_measurements[1]:.2f} {unit}")
        if right_measurements[0] is not None:
            unit = "mm" if tracker.is_calibrated else "px"
            print(f"Right VPF (Vertical Palpebral Height): {right_measurements[0]:.2f} {unit}")
            print(f"Right MRD1 (Margin Reflex Distance 1): {right_measurements[1]:.2f} {unit}")
    else:
        print("No measurements found")
    
    # Display the image with measurements
    cv2.imshow("Improved Eye Measurements", frame)
    print("Displaying image with measurements. Press any key to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the result
    result_dir = os.path.join("eye_tracking_output", "improved_results")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "improved_eye_measurements.jpg")
    cv2.imwrite(result_path, frame)
    print(f"Saved result to: {result_path}")

if __name__ == "__main__":
    test_face_detection() 