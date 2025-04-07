import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import pandas as pd
import ast
import re
from PIL import Image, ImageTk

class EyeTrackingGUI:
    def __init__(self, root, image_folder):
        self.root = root
        self.root.title("Eye Tracking GUI")
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(1000, lambda: self.root.attributes('-topmost', False))

        self.image_folder = image_folder
        self.image_pairs = self.get_eye_image_pairs()
        self.measurements_df = self.load_measurements()

        self.num_pairs = min(len(self.image_pairs), len(self.measurements_df)) if self.measurements_df is not None else len(self.image_pairs)
        self.current_pair_index = 0
        self.current_mode = "vph"

        self.tooltip = tk.Label(self.root, text="", bg="lightyellow", relief="solid", bd=1, font=("Arial", 10))
        self.tooltip.place_forget()

        self.frame_top = tk.Frame(root)
        self.frame_top.grid(row=0, column=0, columnspan=3, pady=5)

        self.frame_images = tk.Frame(root)
        self.frame_images.grid(row=1, column=0, columnspan=3, pady=10)

        self.frame_table_od = tk.Frame(self.frame_images)
        self.frame_table_od.grid(row=0, column=0, padx=10)

        self.left_image_block = tk.Frame(self.frame_images)
        self.left_image_block.grid(row=0, column=1, padx=10)
        self.left_image_label = tk.Label(self.left_image_block)
        self.left_image_label.pack()
        self.left_filename_label = tk.Label(self.left_image_block, text="", font=("Arial", 10, "italic"))
        self.left_filename_label.pack()

        self.right_image_block = tk.Frame(self.frame_images)
        self.right_image_block.grid(row=0, column=2, padx=10)
        self.right_image_label = tk.Label(self.right_image_block)
        self.right_image_label.pack()
        self.right_filename_label = tk.Label(self.right_image_block, text="", font=("Arial", 10, "italic"))
        self.right_filename_label.pack()

        self.frame_table_os = tk.Frame(self.frame_images)
        self.frame_table_os.grid(row=0, column=3, padx=10)

        self.button_frame = tk.Frame(root)
        self.button_frame.grid(row=2, column=0, columnspan=3, pady=5)
        tk.Button(self.button_frame, text="◀ Previous Pair", font=("Arial", 12), command=self.previous_pair).pack(side=tk.LEFT, padx=10)
        tk.Button(self.button_frame, text="Next Pair ▶", font=("Arial", 12), command=self.next_pair).pack(side=tk.LEFT, padx=10)

        self.frame_graph = tk.Frame(root)
        self.frame_graph.grid(row=3, column=1, padx=40)
        self.fig, self.ax = plt.subplots(figsize=(4.5, 2.0))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_graph)
        self.canvas.get_tk_widget().pack()

        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

        self.scrub_frame = tk.Frame(root)
        self.scrub_frame.grid(row=4, column=0, columnspan=3, pady=(0, 10))
        self.scrub_label = tk.Label(self.scrub_frame, text="")
        self.scrub_label.pack()
        self.scrub = tk.Scale(self.scrub_frame, from_=0, to=self.num_pairs - 1, orient=tk.HORIZONTAL, length=600, command=self.scrub_to)
        self.scrub.pack()

        self.toggle_frame = tk.Frame(root)
        self.toggle_frame.grid(row=5, column=0, columnspan=3, pady=10)
        tk.Button(self.toggle_frame, text="Show VPF", font=("Arial", 12), command=lambda: self.set_mode("vph")).pack(side=tk.LEFT, padx=10)
        tk.Button(self.toggle_frame, text="Show MRD1", font=("Arial", 12), command=lambda: self.set_mode("mrd1")).pack(side=tk.LEFT, padx=10)

        self.populate_tables()
        self.create_graph(self.current_mode)
        self.show_images(*self.image_pairs[self.current_pair_index])

    def set_mode(self, mode):
        self.current_mode = mode
        self.create_graph(mode)

    def get_eye_image_pairs(self):
        files = os.listdir(self.image_folder)
        left_images, right_images = {}, {}
        for f in files:
            fname = f.lower()
            if fname.endswith(".jpg"):
                if "left_eye" in fname:
                    key = fname.split("left_eye_")[-1].split(".jpg")[0]
                    left_images[key] = f
                elif "right_eye" in fname:
                    key = fname.split("right_eye_")[-1].split(".jpg")[0]
                    right_images[key] = f
        pairs = [(os.path.join(self.image_folder, left_images[k]), os.path.join(self.image_folder, right_images[k]))
                 for k in sorted(set(left_images) & set(right_images))]
        return pairs

    def load_measurements(self):
        csv_files = [f for f in os.listdir(self.image_folder) if f.endswith(".csv")]
        if not csv_files:
            return None
        return pd.read_csv(os.path.join(self.image_folder, csv_files[0]))

    def populate_tables(self, custom_row=None):
        if len(self.image_pairs) == 0 or self.measurements_df is None or self.measurements_df.empty:
            return
        row = custom_row if custom_row is not None else self.measurements_df.iloc[self.current_pair_index]
        values = {
            "Timestamp": (row.get("timestamp", ""), row.get("timestamp", "")),
            "VPF": (row.get("right_vph", ""), row.get("left_vph", "")),
            "MRD1": (row.get("right_mrd1", ""), row.get("left_mrd1", "")),
            "Blink Rate": ("", ""),
            "TPS": ("", ""),
            "Static HPF": ("", "")
        }
        for frame, eye in zip([self.frame_table_od, self.frame_table_os], ["OD", "OS"]):
            for widget in frame.winfo_children():
                widget.destroy()
            tk.Label(frame, text=f"{eye} Eye", font=("Arial", 16, "bold"), relief=tk.RIDGE, width=20).grid(row=0, column=0, columnspan=2)
            row_counter = 1
            for key, val_pair in values.items():
                val = val_pair[0] if eye == "OD" else val_pair[1]
                if isinstance(val, float):
                    val = round(val, 2)
                tk.Label(frame, text=key, font=("Arial", 14), borderwidth=1, relief="solid", width=15, height=2).grid(row=row_counter, column=0)
                tk.Label(frame, text=val, font=("Arial", 14), borderwidth=1, relief="solid", width=15, height=2).grid(row=row_counter, column=1)
                row_counter += 1

    def create_graph(self, mode="vph"):
        self.ax.clear()
        if self.measurements_df is None or self.measurements_df.empty:
            return

        def eval_fallback(val):
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            if isinstance(val, float) or isinstance(val, int):
                return [val]
            if isinstance(val, str):
                try:
                    return ast.literal_eval(val)
                except:
                    matches = re.findall(r"[-+]?\d*\.\d+|\d+", val)
                    return [float(m) for m in matches]
            return []

        self.left_all, self.right_all, self.timestamps = [], [], []
        for _, row in self.measurements_df.iterrows():
            ts = row.get("timestamp")
            left = eval_fallback(row.get(f"left_{mode}", ""))
            right = eval_fallback(row.get(f"right_{mode}", ""))
            self.left_all.append((ts, sum(left)/len(left) if left else None))
            self.right_all.append((ts, sum(right)/len(right) if right else None))
            self.timestamps.append(ts)

        left_x, left_y = zip(*[(x, y) for x, y in self.left_all if y is not None])
        right_x, right_y = zip(*[(x, y) for x, y in self.right_all if y is not None])

        linestyle_os = (0, (1, 2)) if mode == "vph" else "dotted"
        linestyle_od = (0, (1, 2)) if mode == "vph" else "-"

        self.ax.plot(left_x, left_y, label=f"{mode.upper()} OS", linestyle=linestyle_os, linewidth=1.5, color="#0072B2")
        self.ax.plot(right_x, right_y, label=f"{mode.upper()} OD", linestyle=linestyle_od, linewidth=1.5, color="#E69F00")

        red_index = int((len(self.measurements_df) / self.num_pairs) * self.current_pair_index)
        if red_index < len(self.measurements_df):
            red_ts = self.measurements_df.iloc[red_index].get("timestamp")
            y_left = dict(self.left_all).get(red_ts)
            y_right = dict(self.right_all).get(red_ts)
            if y_left is not None:
                self.ax.plot(red_ts, y_left, 'ro')
            if y_right is not None:
                self.ax.plot(red_ts, y_right, 'ro')

        self.ax.set_xlim(left=min(self.timestamps), right=max(self.timestamps))
        self.ax.set_title(f"{mode.upper()} vs Timestamp", fontname="Arial")
        self.ax.set_xlabel("Timestamp", fontname="Arial")
        self.ax.set_ylabel(mode.upper(), fontname="Arial")
        self.ax.legend(prop={'size': 9})
        self.canvas.draw()

    def on_hover(self, event):
        if event.inaxes == self.ax and event.xdata is not None:
            x = event.xdata
            nearest_ts = min(self.timestamps, key=lambda t: abs(t - x))
            y_left = dict(self.left_all).get(nearest_ts)
            y_right = dict(self.right_all).get(nearest_ts)
            if y_left is not None and y_right is not None:
                hover_row = self.measurements_df.loc[self.measurements_df['timestamp'] == nearest_ts]
                if not hover_row.empty:
                    self.populate_tables(custom_row=hover_row.iloc[0])
                self.tooltip.config(text=f"Timestamp = {nearest_ts:.2f}\nOS = {y_left:.2f}, OD = {y_right:.2f}")
                self.tooltip.place(x=event.guiEvent.x + 10, y=event.guiEvent.y + 10)
        else:
            self.tooltip.place_forget()

    def show_images(self, left_path, right_path):
        try:
            left_img = Image.open(left_path).resize((250, 250))
            right_img = Image.open(right_path).resize((250, 250))
            self.left_imgtk = ImageTk.PhotoImage(left_img)
            self.right_imgtk = ImageTk.PhotoImage(right_img)
            self.left_image_label.config(image=self.left_imgtk)
            self.right_image_label.config(image=self.right_imgtk)
            self.left_filename_label.config(text=os.path.basename(left_path))
            self.right_filename_label.config(text=os.path.basename(right_path))
        except Exception as e:
            print(f"❌ Error displaying images: {e}")

    def update_display(self):
        if len(self.image_pairs) == 0 or self.measurements_df is None or self.measurements_df.empty:
            return
        left_image_file, right_image_file = self.image_pairs[self.current_pair_index]
        self.scrub.set(self.current_pair_index)
        self.scrub_label.config(text=f"{self.current_pair_index}/{self.num_pairs}")
        self.populate_tables()
        self.create_graph(self.current_mode)
        self.show_images(left_image_file, right_image_file)

    def next_pair(self):
        if self.current_pair_index < self.num_pairs - 1:
            self.current_pair_index += 1
            self.update_display()

    def previous_pair(self):
        if self.current_pair_index > 0:
            self.current_pair_index -= 1
            self.update_display()

    def scrub_to(self, val):
        self.current_pair_index = int(val)
        self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    root.update()
    image_folder = filedialog.askdirectory(title="Select Folder Containing Eye Images and CSV")
    if image_folder:
        root.deiconify()
        root.lift()
        root.attributes('-topmost', True)
        root.after(1000, lambda: root.attributes('-topmost', False))
        app = EyeTrackingGUI(root, image_folder)
        root.mainloop()
    else:
        print("No folder selected.")
