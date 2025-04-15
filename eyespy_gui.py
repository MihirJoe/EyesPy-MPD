import tkinter as tk
from tkinter import ttk  # Optional for better widgets
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import os
import random

matplotlib.use("TkAgg")  # Ensure Matplotlib integrates with Tkinter

class EyeTrackingGUI:
    def __init__(self, root, image_folder="./eye_tracking_output/", data_file = None): # Aditi's folder: "/Users/aditi/Desktop/MPD/EyesPy-MPD/examples_for_gui"
        print("Initializing GUI")  # Debugging
        self.root = root
        self.root.title("Eye Tracking GUI")

        # Bring window to the front
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(1000, lambda: self.root.attributes('-topmost', False))

        # Image Folder
        self.image_folder = image_folder
        self.data_file = data_file

        # Load Image Pairs
        self.image_pairs = self.get_eye_image_pairs()
        self.current_pair_index = 0  

        # UI Setup
        self.frame_top = tk.Frame(root)
        self.frame_top.grid(row=0, column=0, columnspan=3, pady=5)
        self.label_filename = tk.Label(self.frame_top, text="File: No image loaded", font=("Arial", 12, "bold"))
        self.label_filename.pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.grid(row=1, column=0, columnspan=3, pady=5)
        self.prev_button = tk.Button(self.button_frame, text="◀ Previous Pair", command=self.load_previous_pair)
        self.prev_button.pack(side="left", padx=10)
        self.next_button = tk.Button(self.button_frame, text="Next Pair ▶", command=self.load_next_pair)
        self.next_button.pack(side="right", padx=10)

        self.frame_od = tk.Frame(root, width=400, height=400)
        self.frame_od.grid(row=2, column=0, padx=10, pady=10)
        self.frame_os = tk.Frame(root, width=400, height=400)
        self.frame_os.grid(row=2, column=2, padx=10, pady=10)
        self.label_od = tk.Label(self.frame_od, text="OD (Right Eye)")
        self.label_od.pack(fill='both')
        self.label_os = tk.Label(self.frame_os, text="OS (Left Eye)")
        self.label_os.pack(fill='both')

        self.frame_reliability = tk.Frame(root, width=150, height=200)
        self.frame_reliability.grid(row=2, column=1, padx=10, pady=10)
        self.label_reliability = tk.Label(self.frame_reliability, text="Reliability\nOD: -- %\nOS: -- %", font=("Arial", 12, "bold"))
        self.label_reliability.pack()

        self.frame_graph = tk.Frame(root, width=600, height=200)
        self.frame_graph.grid(row=3, column=0, columnspan=3, padx=10, pady=10)
        self.create_graph()

        self.frame_table_od = tk.Frame(root, width=300, height=100)
        self.frame_table_od.grid(row=4, column=0, padx=10, pady=10)
        self.frame_table_os = tk.Frame(root, width=300, height=100)
        self.frame_table_os.grid(row=4, column=2, padx=10, pady=10)

        print("Calling populate_tables()")  
        self.populate_tables()

        if self.image_pairs:
            self.load_next_pair()
        else:
            print("⚠️ No valid image pairs found!")

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
        cols = ["Measurement", "Average Value"]
        data = ["VPF", "HPF", "MRD1", "Blink Rate", "Upper Lid Crease"]
        values = {key: round(random.uniform(0.1, 1.0), 2) for key in data}

        for frame in [self.frame_table_od, self.frame_table_os]:
            for i, text in enumerate(cols):
                tk.Label(frame, text=text, relief=tk.RIDGE, width=15, font=("Arial", 10, "bold")).grid(row=0, column=i)

            for i, key in enumerate(data):
                tk.Label(frame, text=key, relief=tk.RIDGE, width=15).grid(row=i + 1, column=0)
                tk.Label(frame, text=str(values[key]), relief=tk.RIDGE, width=15).grid(row=i + 1, column=1)

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
        left_image_file, right_image_file = self.image_pairs[self.current_pair_index]
        
        left_image_path = os.path.join(self.image_folder, left_image_file)
        right_image_path = os.path.join(self.image_folder, right_image_file)
        
        left_image = Image.open(left_image_path)
        right_image = Image.open(right_image_path)
        
        resized_left_image = left_image.resize((400, 400), Image.Resampling.LANCZOS)
        resized_right_image = right_image.resize((400, 400), Image.Resampling.LANCZOS)

        left_photo = ImageTk.PhotoImage(image=resized_left_image)
        right_photo = ImageTk.PhotoImage(image=resized_right_image)

        self.label_od.right_photo = right_photo
        self.label_od.configure(image=right_photo)
        self.label_os.image = left_photo       
        self.label_os.configure(image=left_photo)

        self.label_os.pack(expand=True, fill='both')
        self.label_od.pack(expand=True, fill='both')


        self.label_filename.config(text=f"File: {left_image_file} & {right_image_file}")

    def create_graph(self):
        """Creates the VPF graph."""
        self.fig, self.ax = plt.subplots(figsize=(6, 2))
        self.time = np.linspace(0, 10, 100)
        self.line_os, = self.ax.plot([], [], label="VPF OS", color="blue")
        self.line_od, = self.ax.plot([], [], label="VPF OD", color="green")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("VPF")
        self.ax.legend()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_graph)
        self.canvas.get_tk_widget().pack()

if __name__ == "__main__":
    
    root = tk.Tk(baseName="GUI")
    app = EyeTrackingGUI(root)
    root.mainloop()