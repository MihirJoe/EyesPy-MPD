import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from segment_anything import sam_model_registry, SamPredictor

class EyespyGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Camera and predictor setup
        self.cap = cv2.VideoCapture(0)
        self.pixels_per_mm = None  # Set after calibration

        # Load SAM Predictor
        sam_checkpoint = "sam_vit_h_4b8939.pth"  # Replace with your path
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(sam)

        # GUI setup
        self.init_ui()

        # Timer to update frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS

    def init_ui(self):
        self.setWindowTitle("EyeSpy VPF/MRD1 Measurement")

        self.video_label = QLabel(self)

        self.calibrate_button = QPushButton("Calibrate", self)
        self.calibrate_button.clicked.connect(self.calibrate)

        self.measure_button = QPushButton("Start Measuring", self)
        self.measure_button.clicked.connect(self.start_measuring)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.calibrate_button)
        layout.addWidget(self.measure_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.is_measuring = False

    def calibrate(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not capture frame for calibration.")
            return

        # Crop eye region if you want (for now, full frame)
        eye_crop = frame

        self.predictor.set_image(eye_crop)

        input_point = np.array([[eye_crop.shape[1] // 2, eye_crop.shape[0] // 2]])
        input_label = np.array([1])

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        best_mask = masks[np.argmax(scores)]

        x, y, w, h = cv2.boundingRect(best_mask.astype(np.uint8))
        corneal_diameter_px = w

        self.pixels_per_mm = corneal_diameter_px / 11.8

        print(f"Calibration complete: {self.pixels_per_mm:.2f} pixels per mm")

    def start_measuring(self):
        if self.pixels_per_mm is None:
            print("Error: Please calibrate before measuring!")
            return

        self.is_measuring = True
        print("Measurement started!")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Display frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))

        if self.is_measuring and self.pixels_per_mm is not None:
            # Fake example: let's assume vpf_px and mrd1_px are extracted from the frame
            vpf_px, mrd1_px = self.fake_measurements(frame)

            vpf_mm = vpf_px / self.pixels_per_mm
            mrd1_mm = mrd1_px / self.pixels_per_mm

            print(f"VPF: {vpf_mm:.2f} mm, MRD1: {mrd1_mm:.2f} mm")

    def fake_measurements(self, frame):
        # Dummy placeholder; replace with real VPF and MRD1 extraction
        vpf_px = 40
        mrd1_px = 25
        return vpf_px, mrd1_px

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = EyespyGUI()
    window.show()
    sys.exit(app.exec_())
