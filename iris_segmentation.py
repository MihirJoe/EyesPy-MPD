import os
import cv2
import dlib
import numpy as np
import torch
import glob
from segment_anything import sam_model_registry, SamPredictor

# -------------------- Helper Functions --------------------
def convert_pixels_to_mm(diameter_px, reference_corneal_diameter_mm=11.8, corneal_diameter_px=120):
    """
    Convert pixel diameter to millimeters using a reference corneal size.
    Args:
        diameter_px: Measured iris diameter in pixels
        reference_corneal_diameter_mm: Assumed corneal diameter in mm (typically 11.8mm)
        corneal_diameter_px: Corneal diameter in pixels estimated from images
    Returns:
        diameter_mm: Diameter in millimeters
    """
    pixels_per_mm = corneal_diameter_px / reference_corneal_diameter_mm
    diameter_mm = diameter_px / pixels_per_mm
    return diameter_mm

def find_iris_dark_blob(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest)
    return int(x), int(y), int(radius)

def find_iris_with_hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=gray.shape[0] // 4,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=gray.shape[0] // 2
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        sorted_circles = sorted(circles[0, :], key=lambda c: c[2], reverse=True)
        best_circle = sorted_circles[0]
        center_x, center_y, radius = best_circle
        return center_x, center_y, radius
    else:
        return None

def detect_iris_center_and_radius(image):
    iris_circle = find_iris_with_hough(image)
    if iris_circle is not None:
        return iris_circle
    return find_iris_dark_blob(image)

def is_mask_circular(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter == 0:
        return False

    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return 0.6 < circularity < 1.4
# Helper function to refine mask to darkest blob inside mask, focusing on iris
def refine_mask_to_darkest_blob(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Adaptive threshold instead of fixed 50
    thresh = cv2.adaptiveThreshold(
        masked_gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=5
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    if perimeter == 0:
        return mask
    circularity = 4 * np.pi * (area / (perimeter * perimeter))

    mask_area = mask.shape[0] * mask.shape[1]
    area_ratio = area / mask_area
    if 0.01 < area_ratio < 0.2 and 0.8 < circularity < 1.2:
        refined_mask = np.zeros_like(mask)
        cv2.drawContours(refined_mask, [largest], -1, 255, thickness=cv2.FILLED)
        return refined_mask
    else:
        return mask

def find_dark_region_center(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(blurred)
    return minLoc  # Returns (x, y)

# Helper function to crop the central vertical band of an eye image
def crop_eye_strip(eye_img):
    h, w = eye_img.shape[:2]
    y_center = h // 2
    band_height = int(h * 0.25)  # 25% up and down from center
    y1 = max(y_center - band_height, 0)
    y2 = min(y_center + band_height, h)
    return eye_img[y1:y2, :]

# -------------------- IrisSegmentor Class --------------------
class IrisSegmentor:
    def __init__(self, use_sam=True):
        """
        Initialize the iris segmentation module.
        Args:
            use_sam (bool): Whether to use SAM for iris segmentation
        """
        # Initialize face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # SAM model setup
        self.use_sam = use_sam
        self.sam_predictor = None
        if self.use_sam:
            self.initialize_sam_model()

        # Create output directory
        self.output_dir = "mihir_test"
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize_sam_model(self):
        """Initialize the SAM model for iris segmentation"""
        try:
            # Look for model weights in the current directory first, then in models/
            model_path = None
            # Check current directory for .pth files that contain "sam"
            current_dir_models = glob.glob("*.pth")
            sam_models = [m for m in current_dir_models if "sam" in m.lower()]

            if sam_models:
                model_path = sam_models[0]
                print(f"Found SAM model in current directory: {model_path}")
            else:
                # Check models directory as fallback
                models_dir_path = "models/sam_vit_b_01ec64.pth"
                if os.path.exists(models_dir_path):
                    model_path = models_dir_path

            if not model_path:
                print("SAM model weights not found. Please ensure a .pth file with 'sam' in the name exists.")
                self.use_sam = False
                return False

            # Determine the model type based on the filename
            model_type = "vit_b"  # Default model type
            if "vit_h" in model_path.lower():
                model_type = "vit_h"
            elif "vit_l" in model_path.lower():
                model_type = "vit_l"

            # Initialize the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device=device)
            self.sam_predictor = SamPredictor(sam)
            print(f"SAM model initialized (type: {model_type}) on {device}")
            return True
        except Exception as e:
            print(f"Failed to initialize SAM model: {e}")
            self.use_sam = False
            return False

    def detect_eyes(self, image):
        """
        Detect eyes using facial landmarks.
        Args:
            image: Input image
        Returns:
            Dictionary containing eye data or None if no faces detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            print("No faces detected in the image")
            return None

        face = faces[0]
        landmarks = self.predictor(gray, face)
        left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        padding = 10
        left_eye_x = max(0, min(p[0] for p in left_eye_points) - padding)
        left_eye_y = max(0, min(p[1] for p in left_eye_points) - padding)
        left_eye_w = max(p[0] for p in left_eye_points) - left_eye_x + padding
        left_eye_h = max(p[1] for p in left_eye_points) - left_eye_y + padding
        right_eye_x = max(0, min(p[0] for p in right_eye_points) - padding)
        right_eye_y = max(0, min(p[1] for p in right_eye_points) - padding)
        right_eye_w = max(p[0] for p in right_eye_points) - right_eye_x + padding
        right_eye_h = max(p[1] for p in right_eye_points) - right_eye_y + padding
        h, w = image.shape[:2]
        left_eye_w = min(left_eye_w, w - left_eye_x)
        left_eye_h = min(left_eye_h, h - left_eye_y)
        right_eye_w = min(right_eye_w, w - right_eye_x)
        right_eye_h = min(right_eye_h, h - right_eye_y)
        left_eye_image = image[left_eye_y:left_eye_y+left_eye_h, left_eye_x:left_eye_x+left_eye_w]
        right_eye_image = image[right_eye_y:right_eye_y+right_eye_h, right_eye_x:right_eye_x+right_eye_w]
        left_eye_coords = (left_eye_x, left_eye_y, left_eye_w, left_eye_h)
        right_eye_coords = (right_eye_x, right_eye_y, right_eye_w, right_eye_h)
        return {
            'left_eye': {
                'image': left_eye_image,
                'coords': left_eye_coords
            },
            'right_eye': {
                'image': right_eye_image,
                'coords': right_eye_coords
            },
            'landmarks': landmarks
        }

    def split_eye_image(self, image):
        h, w = image.shape[:2]
        max_width = 800
        max_height = 600
        scale_w = min(1.0, max_width / w)
        scale_h = min(1.0, max_height / h)
        scale = min(scale_w, scale_h)

        display_image = cv2.resize(image, (int(w * scale), int(h * scale)))

        print("Manual eye selection required: Please click LEFT eye center, then RIGHT eye center.")
        clicks = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                true_x = int(x / scale)
                true_y = int(y / scale)
                clicks.append((true_x, true_y))
                print(f"Clicked at (display): ({x}, {y}), mapped to (original): ({true_x}, {true_y})")

        window_name = "Click Left Eye, then Right Eye"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, click_event)

        while True:
            cv2.imshow(window_name, display_image)
            key = cv2.waitKey(1) & 0xFF
            if len(clicks) >= 2:
                break
            if key == 27:  # ESC to cancel
                print("Eye click cancelled.")
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()

        def crop_eye(img, center):
            cx, cy = center
            box_size = int(min(img.shape[0], img.shape[1]) * 0.3)
            x1 = max(cx - box_size // 2, 0)
            y1 = max(cy - box_size // 2, 0)
            x2 = min(cx + box_size // 2, img.shape[1])
            y2 = min(cy + box_size // 2, img.shape[0])
            return img[y1:y2, x1:x2]

        left_eye_center, right_eye_center = clicks[0], clicks[1]

        left_eye_image = crop_eye(image, left_eye_center)
        right_eye_image = crop_eye(image, right_eye_center)

        right_eye_coords = (0, 0, right_eye_image.shape[1], right_eye_image.shape[0])
        left_eye_coords = (0, 0, left_eye_image.shape[1], left_eye_image.shape[0])

        print(f"Split image into left eye ({left_eye_image.shape[:2]}) and right eye ({right_eye_image.shape[:2]})")
        cv2.imwrite(os.path.join(self.output_dir, "split_left_eye.png"), left_eye_image)
        cv2.imwrite(os.path.join(self.output_dir, "split_right_eye.png"), right_eye_image)

        return {
            'left_eye': {
                'image': left_eye_image,
                'coords': left_eye_coords
            },
            'right_eye': {
                'image': right_eye_image,
                'coords': right_eye_coords
            }
        }

    def segment_iris_with_sam(self, eye_image, eye_side="left"):
        if not self.use_sam or eye_image is None or eye_image.size == 0:
            return None, None, None
        try:
            h, w = eye_image.shape[:2]
            iris_circle = detect_iris_center_and_radius(eye_image)
            if iris_circle is None:
                print("No iris detected")
                return None, None, None
            center_x, center_y, radius = iris_circle
            pad = int(radius * 0.6)
            x1 = max(center_x - pad, 0)
            y1 = max(center_y - pad, 0)
            x2 = min(center_x + pad, w)
            y2 = min(center_y + pad, h)
            bbox = np.array([x1, y1, x2, y2])
            center_point = np.array([[center_x, center_y]])
            rgb_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(rgb_image)
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=center_point,
                point_labels=np.array([1]),
                box=bbox[np.newaxis, :],
                multimask_output=True
            )
            best_mask_idx = np.argmax(scores)
            iris_mask = masks[best_mask_idx]
            iris_mask_uint8 = (iris_mask * 255).astype(np.uint8)
            iris_mask_uint8 = refine_mask_to_darkest_blob(eye_image, iris_mask_uint8)
            if not is_mask_circular(iris_mask_uint8):
                print("Mask rejected: not circular enough")
                return None, None, None
            contours, _ = cv2.findContours(iris_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return iris_mask_uint8, None, None
            largest_contour = max(contours, key=cv2.contourArea)
            # Prefer ellipse fitting for better center/size accuracy; fall back to circle if not enough points
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                (cx, cy), (major_axis, minor_axis), angle = ellipse
                diameter = (major_axis + minor_axis) / 2  # Approximate average diameter
                return iris_mask_uint8, diameter, (int(cx), int(cy))
            else:
                # fallback to minEnclosingCircle if not enough points
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                diameter = radius * 2
                return iris_mask_uint8, diameter, (int(cx), int(cy))
        except Exception as e:
            print(f"Error in SAM iris segmentation: {e}")
            return None, None, None

    def process_image(self, image_path, visualize=True):
        """
        Process an image to detect eyes and segment the iris.
        Args:
            image_path: Path to the input image
            visualize: Whether to create visualization images
        Returns:
            results: Dictionary containing results for both eyes
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        eye_data = self.detect_eyes(image)
        if eye_data is None:
            print("No face detected. Trying to split the image into left and right eyes...")
            eye_data = self.split_eye_image(image)
            eye_keys = ['left_eye', 'right_eye']
        else:
            eye_keys = ['left_eye', 'right_eye']
        results = {}
        for key in eye_keys:
            results[key] = {}
        for eye_key in eye_keys:
            eye_image = eye_data[eye_key]['image']
            eye_coords = eye_data[eye_key]['coords']
            iris_mask, iris_diameter, iris_center = self.segment_iris_with_sam(eye_image, eye_side=eye_key)
            if iris_mask is not None:
                mask_filename = os.path.join(self.output_dir, f"{eye_key}_iris_mask.png")
                cv2.imwrite(mask_filename, iris_mask)
                results[eye_key]['mask_path'] = mask_filename
                results[eye_key]['diameter'] = iris_diameter
                iris_diameter_mm = convert_pixels_to_mm(iris_diameter)
                results[eye_key]['diameter_mm'] = iris_diameter_mm
                if visualize:
                    vis_image = eye_image.copy()
                    if iris_mask.shape[:2] != eye_image.shape[:2]:
                        print(f"Warning: Mask shape {iris_mask.shape[:2]} doesn't match image shape {eye_image.shape[:2]}")
                        iris_mask = cv2.resize(iris_mask, (eye_image.shape[1], eye_image.shape[0]))
                    color_mask = np.zeros_like(eye_image)
                    color_mask[iris_mask > 0] = [0, 255, 0]  # Green mask
                    alpha = 0.5
                    blended = cv2.addWeighted(vis_image, 1, color_mask, alpha, 0)
                    
                    # Draw blue circle for iris diameter
                    if iris_diameter is not None and iris_center is not None:
                        center_x, center_y = iris_center
                        radius = int(iris_diameter / 2)
                        cv2.circle(blended, (center_x, center_y), radius, (255, 0, 0), 2)

                    vis_filename = os.path.join(self.output_dir, f"{eye_key}_visualization.png")
                    cv2.imwrite(vis_filename, blended)
                    results[eye_key]['visualization_path'] = vis_filename
        return results

# -------------------- Main Entrypoint --------------------
def main():
    segmentor = IrisSegmentor()
    image_path = input("Enter the path to the input image: ")
    results = segmentor.process_image(image_path)
    if results:
        print("Processing completed successfully.")
        print("Results saved to directory:", segmentor.output_dir)
        for eye_key, eye_data in results.items():
            if 'diameter' in eye_data:
                print(f"{eye_key.replace('_', ' ').title()} iris diameter: {eye_data['diameter']:.2f} pixels ({eye_data['diameter_mm']:.2f} mm)")
    else:
        print("Error processing the image.")

if __name__ == "__main__":
    main()