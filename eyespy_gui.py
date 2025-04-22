import cv2
from segment_anything import sam_model_registry, SamPredictor
import tkinter as tk
from tkinter import ttk  # For better widgets including scrollbars
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import os
import random

matplotlib.use("TkAgg")  # Ensure Matplotlib integrates with Tkinter

class ScrollableFrame(ttk.Frame):
    """A scrollable frame that can be used when the window is too small"""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

class EyeTrackingGUI:
    def __init__(self, root, image_folder="./eye_tracking_output/", data_file = None): # Aditi's folder: "/Users/aditi/Desktop/MPD/EyesPy-MPD/examples_for_gui"
        print("Initializing GUI")  # Debugging
        self.root = root
        self.root.title("Eye Tracking GUI")
        
        # Configure window size and screen positioning
        # Get screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Set window size to 70% of screen dimensions (reduced from 85%)
        window_width = int(screen_width * 0.7)
        window_height = int(screen_height * 0.7)
        
        # Calculate position for center of screen
        center_x = int((screen_width - window_width) / 2)
        center_y = int((screen_height - window_height) / 2)
        
        # Set window size and position
        self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        
        # Set minimum size to ensure UI elements don't get too cramped
        min_width = min(800, int(screen_width * 0.6))
        min_height = min(600, int(screen_height * 0.6))
        self.root.minsize(min_width, min_height)
        
        # Make window resizable
        self.root.resizable(True, True)
        
        # Configure grid weight to make it responsive
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1) 
        root.grid_columnconfigure(2, weight=1)
        for i in range(5):
            root.grid_rowconfigure(i, weight=1)

        # Bring window to the front
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(1000, lambda: self.root.attributes('-topmost', False))

        # Image Folder - Strip any quotes and ensure it exists
        self.image_folder = image_folder.strip("'\"") if image_folder else "./eye_tracking_output/"
        # Ensure the path ends with a separator
        if not self.image_folder.endswith(os.path.sep):
            self.image_folder = self.image_folder + os.path.sep
            
        self.data_file = data_file
        self.pixels_per_mm = None
        self.cap = cv2.VideoCapture(0)

        sam_checkpoint = "sam_vit_b_01ec64.pth"  # Update this to your SAM checkpoint path
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(sam)

        # Load Image Pairs
        self.image_pairs = self.get_eye_image_pairs()
        self.current_pair_index = 0  

        # Calculate image size based on window size - reduced proportions
        self.image_width = int(window_width * 0.25)  # 25% of window width
        self.image_height = int(window_height * 0.25) # 25% of window height

        # Create main content frame (potentially scrollable)
        self.use_scrollable = window_height < 600  # Use scrollable frame if window is small
        
        if self.use_scrollable:
            self.main_frame = ScrollableFrame(root)
            self.main_frame.pack(fill="both", expand=True)
            content_frame = self.main_frame.scrollable_frame
        else:
            content_frame = root

        # UI Setup with more compact layout
        self.frame_top = tk.Frame(content_frame)
        self.frame_top.grid(row=0, column=0, columnspan=3, pady=5, sticky="ew")
        self.label_filename = tk.Label(self.frame_top, text="File: No image loaded", font=("Arial", 11, "bold"))
        self.label_filename.pack()

        self.button_frame = tk.Frame(content_frame)
        self.button_frame.grid(row=1, column=0, columnspan=3, pady=2, sticky="ew")
        self.prev_button = tk.Button(self.button_frame, text="◀ Previous", command=self.load_previous_pair)
        self.prev_button.pack(side="left", padx=10)
        self.next_button = tk.Button(self.button_frame, text="Next ▶", command=self.load_next_pair)
        self.next_button.pack(side="right", padx=10)
        self.calibrate_button = tk.Button(self.button_frame, text="Calibrate", command=self.calibrate)
        self.calibrate_button.pack(side="left", padx=10)
    def calibrate(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not capture frame for calibration.")
            return

        # Optional: crop tight around the eye if you want, for now use full frame
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

        # Image frames with calculated sizes
        self.frame_od = tk.Frame(content_frame, width=self.image_width, height=self.image_height)
        self.frame_od.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        self.frame_od.grid_propagate(False)  # Prevent frame from resizing to contents
        
        self.frame_os = tk.Frame(content_frame, width=self.image_width, height=self.image_height)
        self.frame_os.grid(row=2, column=2, padx=5, pady=5, sticky="nsew")
        self.frame_os.grid_propagate(False)  # Prevent frame from resizing to contents
        
        self.label_od_title = tk.Label(self.frame_od, text="OD (Right Eye)")
        self.label_od_title.pack()
        self.label_od = tk.Label(self.frame_od)
        self.label_od.pack(expand=True, fill="both", padx=2, pady=2)
        
        self.label_os_title = tk.Label(self.frame_os, text="OS (Left Eye)")
        self.label_os_title.pack()
        self.label_os = tk.Label(self.frame_os)
        self.label_os.pack(expand=True, fill="both", padx=2, pady=2)

        # Center reliability frame
        self.frame_reliability = tk.Frame(content_frame)
        self.frame_reliability.grid(row=2, column=1, padx=5, pady=5, sticky="nsew")
        self.label_reliability = tk.Label(self.frame_reliability, text="Reliability\nOD: -- %\nOS: -- %", font=("Arial", 11, "bold"))
        self.label_reliability.pack(expand=True)

        # Reduced height for graph
        self.frame_graph = tk.Frame(content_frame)
        self.frame_graph.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        self.create_graph()

        # Measurement tables
        self.frame_table_od = tk.Frame(content_frame)
        self.frame_table_od.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")
        self.frame_table_os = tk.Frame(content_frame)
        self.frame_table_os.grid(row=4, column=2, padx=5, pady=5, sticky="nsew")

        print("Calling populate_tables()")  
        self.populate_tables()

        # Bind resize event to update image sizes
        self.root.bind("<Configure>", self.on_window_resize)

        if self.image_pairs:
            self.load_next_pair()
        else:
            print("⚠️ No valid image pairs found!")

    def on_window_resize(self, event):
        """Handle window resize events"""
        # Only respond to window resizing, not other configure events
        if event.widget == self.root:
            # Update image dimensions based on new window size
            if hasattr(self, 'last_width') and hasattr(self, 'last_height'):
                if self.last_width == self.root.winfo_width() and self.last_height == self.root.winfo_height():
                    return  # No actual size change
                    
            self.last_width = self.root.winfo_width()
            self.last_height = self.root.winfo_height()
            
            # Recalculate image size - reduced proportions
            self.image_width = int(self.last_width * 0.25)  # Reduced from 30%
            self.image_height = int(self.last_height * 0.25) # Reduced from 30%
            
            # Update frame sizes
            self.frame_od.config(width=self.image_width, height=self.image_height)
            self.frame_os.config(width=self.image_width, height=self.image_height)
            
            # If images are loaded, refresh them with new size
            if self.image_pairs and hasattr(self, 'current_pair_index'):
                self.display_images()

    def get_eye_image_pairs(self):
        """Finds all valid left and right eye pairs in the directory."""
        print("Getting eye image pairs...")  # Debugging

        if not os.path.exists(self.image_folder):
            print("⚠️ Error: Image folder does not exist!")
            return []

        files = sorted(os.listdir(self.image_folder))
        pairs = {}

        for file in files:
            if "left_eye" in file:
                subject_id = file.replace("left_eye", "")
                right_eye_file = file.replace("left_eye", "right_eye")
                if right_eye_file in files:
                    pairs[subject_id] = (file, right_eye_file)

        if not pairs:
            print("⚠️ No valid image pairs found in the folder.")

        return list(pairs.values()) 

    def populate_tables(self):
        """Populates measurement tables with random values."""
        print("Populating tables...")  
        cols = ["Measure", "Value"]  # Shorter column header
        data = ["VPF", "HPF", "MRD1", "Blink", "Lid"]  # Shorter labels
        values = {key: round(random.uniform(0.1, 1.0), 2) for key in data}

        for frame in [self.frame_table_od, self.frame_table_os]:
            for i, text in enumerate(cols):
                tk.Label(frame, text=text, relief=tk.RIDGE, width=8, font=("Arial", 9, "bold")).grid(row=0, column=i)  # Reduced width and font size

            for i, key in enumerate(data):
                tk.Label(frame, text=key, relief=tk.RIDGE, width=8).grid(row=i + 1, column=0)  # Reduced width
                tk.Label(frame, text=str(values[key]), relief=tk.RIDGE, width=8).grid(row=i + 1, column=1)  # Reduced width

    def load_previous_pair(self):
        """Loads the previous image pair."""
        if self.image_pairs:
            self.current_pair_index = (self.current_pair_index - 1) % len(self.image_pairs)
            self.display_images()

    def load_next_pair(self):
        """Loads the next image pair."""
        if self.image_pairs:
            self.current_pair_index = (self.current_pair_index + 1) % len(self.image_pairs)
            self.display_images()

    def display_images(self):
        """Displays the current pair of images."""
        if not self.image_pairs:
            return
            
        left_image_file, right_image_file = self.image_pairs[self.current_pair_index]
        
        left_image_path = os.path.join(self.image_folder, left_image_file)
        right_image_path = os.path.join(self.image_folder, right_image_file)
        
        try:
            # Load the original images
            left_image = Image.open(left_image_path)
            right_image = Image.open(right_image_path)
            
            # Get original aspect ratios
            left_aspect = left_image.width / left_image.height
            right_aspect = right_image.width / right_image.height
            
            # Adjust size to maintain aspect ratio and fit within container
            def fit_image_to_container(img, max_width, max_height, original_aspect):
                # Determine if width or height should be the limiting factor
                if max_width / max_height > original_aspect:
                    # Height limited
                    new_height = max_height
                    new_width = int(max_height * original_aspect)
                else:
                    # Width limited
                    new_width = max_width
                    new_height = int(max_width / original_aspect)
                
                # Resize image
                return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Resize with aspect ratio preservation
            left_image = fit_image_to_container(left_image, self.image_width, self.image_height, left_aspect)
            right_image = fit_image_to_container(right_image, self.image_width, self.image_height, right_aspect)

            # Create PhotoImage objects
            left_photo = ImageTk.PhotoImage(left_image)
            right_photo = ImageTk.PhotoImage(right_image)

            # Update the labels
            self.label_od.config(image=right_photo)
            self.label_od.image = right_photo  # Keep reference to prevent garbage collection
            self.label_os.config(image=left_photo)
            self.label_os.image = left_photo  # Keep reference to prevent garbage collection

            self.label_filename.config(text=f"File: {left_image_file} & {right_image_file}")
            
            # Add frames to properly contain images
            self.label_od.config(width=self.image_width, height=self.image_height)
            self.label_os.config(width=self.image_width, height=self.image_height)
            
        except Exception as e:
            print(f"Error loading images: {e}")
            self.label_filename.config(text=f"Error loading images: {e}")

    def create_graph(self):
        """Creates the VPF graph."""
        self.fig, self.ax = plt.subplots(figsize=(4, 1))  # Further reduced graph size
        self.time = np.linspace(0, 10, 100)
        self.line_os, = self.ax.plot([], [], label="OS", color="blue")  # Shorter label
        self.line_od, = self.ax.plot([], [], label="OD", color="green")  # Shorter label
        self.ax.set_xlabel("Time", fontsize=8)  # Smaller font
        self.ax.set_ylabel("VPF", fontsize=8)  # Smaller font
        self.ax.tick_params(axis='both', which='major', labelsize=7)  # Smaller tick labels
        self.ax.legend(loc='upper right', fontsize='x-small')  # Even smaller legend
        self.fig.tight_layout()  # Optimize layout
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_graph)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = EyeTrackingGUI(root)
    root.mainloop()