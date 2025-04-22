import cv2
import numpy as np
import os
from left_and_right_eye_measurements import EyeTracker

# Constant to represent the original eye image from the query
EXAMPLE_EYE_IMAGE = """
This is a test script to analyze the eye image provided in the user query.
The image shows a close-up of an eye with a green bounding box around it.
There appears to be measurements for VPF (Vertical Palpebral Fissure) and MRD1 (Margin Reflex Distance 1).
"""

def process_image(image):
    """Process an image and show measurements"""
    # Create a tracker with testing enabled (for more verbose output)
    tracker = EyeTracker(file=0, saving=True, testing=True, use_calibration=False)
    
    # Process the frame - this will use our improved eye measurements
    frame, left_measurements, right_measurements = tracker.get_eye_measurements(image)
    
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
    output_path = os.path.join(output_dir, f"processed_eye_image.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Processed image saved to: {output_path}")
    
    return frame

def main():
    print("\nLooking for videos or images in the workspace...")
    
    # Look for video files in the workspace that might contain faces
    video_extensions = ['.mp4', '.mov', '.avi']
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    potential_files = []
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions + image_extensions):
                potential_files.append(os.path.join(root, file))
    
    if not potential_files:
        print("No suitable video or image files found. Creating a test image...")
        face_img = create_test_face()
        process_image(face_img)
        return
    
    # Print found files
    print(f"Found {len(potential_files)} potential files:")
    for i, file in enumerate(potential_files):
        print(f"{i+1}. {file}")
    
    # Try the first image file or extract frame from video
    for file_path in potential_files:
        if any(file_path.lower().endswith(ext) for ext in image_extensions):
            print(f"\nProcessing image file: {file_path}")
            img = cv2.imread(file_path)
            if img is not None:
                process_image(img)
                return
        elif any(file_path.lower().endswith(ext) for ext in video_extensions):
            print(f"\nExtracting frame from video file: {file_path}")
            try:
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    process_image(frame)
                    return
            except Exception as e:
                print(f"Error extracting frame from video: {e}")
    
    print("Could not process any files. Creating a test image...")
    face_img = create_test_face()
    process_image(face_img)

def create_test_face():
    """Create a synthetic face image for testing"""
    # Create a face image with eye visible
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
    
    # Make the face more realistic for face detection
    # Add more detailed features
    cv2.circle(face_img, (175, 200), 3, (255, 255, 255), -1)  # Eye highlights
    cv2.circle(face_img, (325, 200), 3, (255, 255, 255), -1)
    
    return face_img

if __name__ == "__main__":
    main() 