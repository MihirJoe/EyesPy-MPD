import os
import cv2
import dlib
import numpy as np
import torch
import glob
from segment_anything import build_sam, SamPredictor  # Import Iris-SAM custom loader if different

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
    # Bypass strict circularity check; always return True for SAM mask acceptance.
    return True
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
    # Relaxed circularity threshold from (0.8, 1.2) to (0.3, 1.5)
    if 0.01 < area_ratio < 0.2 and 0.3 < circularity < 1.5:
        refined_mask = np.zeros_like(mask)
        cv2.drawContours(refined_mask, [largest], -1, 255, thickness=cv2.FILLED)
        return refined_mask
    else:
        return mask

def find_dark_region_center(image):
    h, w = image.shape[:2]
    band = image[h // 4: 3 * h // 4, :]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, _, minLoc, _ = cv2.minMaxLoc(blurred)
    return (minLoc[0], minLoc[1] + h // 4)

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
        """
        Initialize the default SAM model for iris segmentation.
        """
        try:
            from segment_anything import sam_model_registry
            model_type = "vit_h"
            model_path = "weights/sam_vit_h_4b8939.pth"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device)
            self.sam_predictor = SamPredictor(sam)
            print(f"Default SAM model loaded from {model_path} on {device}")
            return True
        except Exception as e:
            print(f"Failed to load default SAM model: {e}")
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

        # Calculate center of each eye
        cx_left = int(sum(p[0] for p in left_eye_points) / len(left_eye_points))
        cy_left = int(sum(p[1] for p in left_eye_points) / len(left_eye_points))
        cx_right = int(sum(p[0] for p in right_eye_points) / len(right_eye_points))
        cy_right = int(sum(p[1] for p in right_eye_points) / len(right_eye_points))

        # Define fixed-size bounding box (128x128) around each eye center
        box_size = 128  # Tighter crop around the eye
        half_box = box_size // 2
        h, w = image.shape[:2]

        left_eye_x1 = max(cx_left - half_box, 0)
        left_eye_y1 = max(cy_left - half_box, 0)
        left_eye_x2 = min(cx_left + half_box, w)
        left_eye_y2 = min(cy_left + half_box, h)

        right_eye_x1 = max(cx_right - half_box, 0)
        right_eye_y1 = max(cy_right - half_box, 0)
        right_eye_x2 = min(cx_right + half_box, w)
        right_eye_y2 = min(cy_right + half_box, h)

        left_eye_image = image[left_eye_y1:left_eye_y2, left_eye_x1:left_eye_x2]
        right_eye_image = image[right_eye_y1:right_eye_y2, right_eye_x1:right_eye_x2]

        left_eye_coords = (left_eye_x1, left_eye_y1, left_eye_x2 - left_eye_x1, left_eye_y2 - left_eye_y1)
        right_eye_coords = (right_eye_x1, right_eye_y1, right_eye_x2 - right_eye_x1, right_eye_y2 - right_eye_y1)

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
            box_size = 160
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
            return None, None, None, False
        try:
            h, w = eye_image.shape[:2]
            iris_circle = detect_iris_center_and_radius(eye_image)
            if iris_circle is None:
                print("No iris detected")
                return None, None, None, False
            center_x, center_y, radius = iris_circle
            pad = int(radius * 1.6)
            x1 = max(center_x - pad, 0)
            y1 = max(center_y - pad, 0)
            x2 = min(center_x + pad, w)
            y2 = min(center_y + pad, h)
            bbox = np.array([x1, y1, x2, y2])

            # Define a point prompt at the darkest region
            center_px, center_py = find_dark_region_center(eye_image)
            point_coords = np.array([[center_px, center_py]])
            point_labels = np.array([1])  # 1 = foreground

            # CLAHE-based contrast enhancement before SAM
            lab = cv2.cvtColor(eye_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge((l, a, b))
            eye_image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            rgb_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(rgb_image)

            # SAM demo-style: use both box and point prompt, multimask_output=True
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=bbox[np.newaxis, :],
                multimask_output=True
            )
            gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
            best_mask = None
            best_score = float('inf')
            image_center = np.array([w // 2, h // 2])

            for mask_candidate in masks:
                mask_uint8 = (mask_candidate * 255).astype(np.uint8)

                # Area ratio filter
                white_ratio = np.sum(mask_uint8 > 0) / (mask_uint8.shape[0] * mask_uint8.shape[1])
                if white_ratio < 0.01 or white_ratio > 0.8:
                    print(f"Skipping mask due to area ratio: {white_ratio:.2f}")
                    continue

                # Apply a central band mask to focus scoring within horizontal iris band
                vertical_mask = np.zeros_like(mask_uint8)
                h_band = int(mask_uint8.shape[0] * 0.7)
                y_start = int((mask_uint8.shape[0] - h_band) / 2)
                vertical_mask[y_start:y_start + h_band, :] = 1
                banded_mask = cv2.bitwise_and(mask_uint8, mask_uint8, mask=vertical_mask)

                masked_pixels = gray_eye[banded_mask > 0]
                if masked_pixels.size == 0:
                    continue

                mean_gray = np.mean(masked_pixels)

                # Print mean gray and masked pixel count for each candidate
                print(f"Candidate mask mean gray: {mean_gray:.2f}, masked pixels: {masked_pixels.size}")

                contours, _ = cv2.findContours(banded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center_dist = np.linalg.norm(image_center - np.array([cx, cy]))

                area = cv2.contourArea(largest)
                perimeter = cv2.arcLength(largest, True)
                circularity = 0
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))

                if area < 300 or area > (0.5 * h * w):
                    continue

                # Sharpness (variance of Laplacian)
                laplacian = cv2.Laplacian(gray_eye, cv2.CV_64F)
                sharpness = np.var(laplacian[banded_mask > 0])

                # Modified score calculation
                score = 0.5 * mean_gray + 0.3 * center_dist + 50 * abs(circularity - 1) - 0.1 * sharpness

                if score < best_score:
                    best_score = score
                    best_mask = mask_uint8


            iris_mask_uint8 = best_mask if best_mask is not None else (masks[0] * 255).astype(np.uint8)
            masked_pixels = gray_eye[iris_mask_uint8 > 0]
            # Accept the mask if it covers a sufficiently dark region (relaxed threshold)
            if masked_pixels.size > 0 and np.mean(masked_pixels) < 150:
                contours, _ = cv2.findContours(iris_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    return iris_mask_uint8, None, None, False
                largest_contour = max(contours, key=cv2.contourArea)
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    (cx, cy), (major_axis, minor_axis), angle = ellipse
                    diameter = (major_axis + minor_axis) / 2
                    return iris_mask_uint8, diameter, (int(cx), int(cy)), False
                else:
                    (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                    diameter = radius * 2
                    return iris_mask_uint8, diameter, (int(cx), int(cy)), False
            else:
                print("SAM mask does not sufficiently overlap a dark iris region, falling back to Hough or dark blob detection.")
                fallback = detect_iris_center_and_radius(eye_image)
                if fallback is not None:
                    center_x, center_y, radius = fallback
                    diameter = radius * 2
                    return None, diameter, (int(center_x), int(center_y)), True
                return None, None, None, False
        except Exception as e:
            print(f"Error in SAM iris segmentation: {e}")
            return None, None, None, False

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
            iris_mask, iris_diameter, iris_center, used_fallback = self.segment_iris_with_sam(eye_image, eye_side=eye_key)
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
                    # Only overlay green SAM mask if not using fallback
                    if not used_fallback:
                        color_mask = np.zeros_like(eye_image)
                        color_mask[iris_mask > 0] = [0, 255, 0]  # Green mask
                        alpha = 0.5
                        blended = cv2.addWeighted(vis_image, 1, color_mask, alpha, 0)
                    else:
                        blended = vis_image.copy()
                    # Draw blue circle for iris diameter, even if mask isn't perfectly circular
                    if iris_diameter is not None and iris_center is not None:
                        center_x, center_y = iris_center
                        radius = int(iris_diameter / 2)
                        cv2.circle(blended, (center_x, center_y), radius, (255, 0, 0), 2)
                    # Draw the red point used for the prompt
                    dark_x, dark_y = find_dark_region_center(eye_image)
                    cv2.circle(blended, (dark_x, dark_y), 3, (0, 0, 255), -1)  # Red dot

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