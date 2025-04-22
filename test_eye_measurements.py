import cv2
import numpy as np
import os
import base64
from left_and_right_eye_measurements import EyeTracker

def process_image(image_path):
    """Process a single image and show measurements"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Create a tracker with testing enabled (for more verbose output)
    tracker = EyeTracker(file=0, saving=True, testing=True, use_calibration=False)
    
    # Process the frame - this will use our improved eye measurements
    frame, left_measurements, right_measurements = tracker.get_eye_measurements(img)
    
    # Show results
    if left_measurements[0] is not None or right_measurements[0] is not None:
        print("Measurements:")
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
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save output image
    output_dir = os.path.join("eye_tracking_output", "test_results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"processed_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, frame)
    print(f"Processed image saved to: {output_path}")

def save_images_from_input():
    """Create test images from the user input"""
    # Create output directory
    output_dir = os.path.join("eye_tracking_output", "input_images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a face image with eye visible
    face_image_path = os.path.join(output_dir, "face_with_eye.jpg")
    face_img = np.zeros((500, 500, 3), dtype=np.uint8)
    face_img[:] = (220, 210, 200)  # Light skin tone background
    
    # Draw face outline
    cv2.ellipse(face_img, (250, 250), (150, 200), 0, 0, 360, (200, 180, 170), -1)
    
    # Draw eyes
    # Left eye
    cv2.ellipse(face_img, (175, 200), (40, 20), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(face_img, (175, 200), (15, 15), 0, 0, 360, (50, 50, 100), -1)
    cv2.ellipse(face_img, (175, 200), (5, 5), 0, 0, 360, (0, 0, 0), -1)
    
    # Right eye
    cv2.ellipse(face_img, (325, 200), (40, 20), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(face_img, (325, 200), (15, 15), 0, 0, 360, (50, 50, 100), -1)
    cv2.ellipse(face_img, (325, 200), (5, 5), 0, 0, 360, (0, 0, 0), -1)
    
    # Draw eyebrows
    cv2.ellipse(face_img, (175, 170), (45, 10), 0, 0, 180, (80, 70, 60), 4)
    cv2.ellipse(face_img, (325, 170), (45, 10), 0, 0, 180, (80, 70, 60), 4)
    
    # Draw nose and mouth
    cv2.line(face_img, (250, 220), (250, 280), (150, 130, 120), 3)
    cv2.ellipse(face_img, (250, 320), (60, 20), 0, 0, 180, (150, 100, 100), 3)
    
    # Save the image
    cv2.imwrite(face_image_path, face_img)
    print(f"Created test face image at {face_image_path}")
    
    return face_image_path

def main():
    # Create test images from input
    face_image_path = save_images_from_input()
    
    # Process the image
    print("\nProcessing face image...")
    process_image(face_image_path)

if __name__ == "__main__":
    main() 