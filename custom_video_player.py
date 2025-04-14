import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import threading
import time
import os
import pandas as pd

class CustomVideoPlayer:
    """A compact video player widget for Tkinter that fits within the screen"""
    
    def __init__(self, parent_widget, video_source=0, width=None, height=None, measurements_data=None, playback_speed=1.0):
        self.parent = parent_widget
        self.video_source = video_source
        self.measurements_data = measurements_data
        self.playback_speed = playback_speed  # Playback speed multiplier
        
        # Initialize video capture
        self.cap = None
        self.connect_to_video_source()
        
        if self.cap is None or not self.cap.isOpened():
            raise ValueError(f"Unable to open video source {video_source}")
        
        # Get actual video dimensions
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate dimensions that maintain aspect ratio
        aspect_ratio = actual_width / actual_height
        
        # If width and height were provided, use them as constraints
        if width and height:
            available_width = width
            available_height = height
        else:
            # Otherwise use parent dimensions
            # Get parent widget dimensions for scaling
            parent_width = self.parent.winfo_width()
            parent_height = self.parent.winfo_height()
            
            # If parent dimensions aren't available yet, use reasonable defaults
            if parent_width <= 1:
                parent_width = 800
            if parent_height <= 1:
                parent_height = 600
                
            # Use 80% of parent widget dimensions to allow for controls
            available_width = int(parent_width * 0.8)
            available_height = int(parent_height * 0.6)  # Reduced to make more room for controls
        
        # Calculate dimensions that preserve aspect ratio and maximize space
        if available_width / available_height > aspect_ratio:
            # Available space is wider than video
            self.height = available_height
            self.width = int(available_height * aspect_ratio)
        else:
            # Available space is taller than video
            self.width = available_width
            self.height = int(available_width / aspect_ratio)
        
        # Get other video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Cap the FPS at 60 for display purposes if it's very high (slow motion videos)
        self.display_fps = min(self.fps, 60.0) * self.playback_speed
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_num = 0
        
        # Print video info for debugging
        print(f"Video info: FPS={self.fps}, Frames={self.total_frames}, Size={actual_width}x{actual_height}")
        
        # Create frame for video display (will contain canvas)
        self.video_frame = tk.Frame(parent_widget, width=self.width, height=self.height)
        self.video_frame.pack(expand=True, fill=tk.BOTH, pady=5)
        self.video_frame.pack_propagate(False)  # Prevent resizing
        
        # Center the canvas in the frame
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)
        
        # Create canvas for video display - exact size of video
        self.canvas = tk.Canvas(self.video_frame, width=self.width, height=self.height, 
                              bg="black", highlightthickness=0)  # Remove border
        self.canvas.grid(row=0, column=0, sticky="nsew")  # Center in frame
        
        # Create control frame
        self.control_frame = tk.Frame(parent_widget)
        self.control_frame.pack(fill=tk.X, expand=False, pady=5)
        
        # Create video controls
        self.btn_play = tk.Button(self.control_frame, text="▶ Play", command=self.play)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        self.btn_pause = tk.Button(self.control_frame, text="⏸ Pause", command=self.pause)
        self.btn_pause.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = tk.Button(self.control_frame, text="⏹ Stop", command=self.stop)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        # Create speed control buttons
        self.speed_frame = tk.Frame(self.control_frame)
        self.speed_frame.pack(side=tk.LEFT, padx=10)
        
        speed_label = tk.Label(self.speed_frame, text="Speed:")
        speed_label.pack(side=tk.LEFT)
        
        self.btn_speed_slow = tk.Button(self.speed_frame, text="0.5x", command=lambda: self.set_playback_speed(0.5))
        self.btn_speed_slow.pack(side=tk.LEFT, padx=2)
        
        self.btn_speed_normal = tk.Button(self.speed_frame, text="1x", command=lambda: self.set_playback_speed(1.0))
        self.btn_speed_normal.pack(side=tk.LEFT, padx=2)
        
        self.btn_speed_fast = tk.Button(self.speed_frame, text="2x", command=lambda: self.set_playback_speed(2.0))
        self.btn_speed_fast.pack(side=tk.LEFT, padx=2)
        
        # Create time slider
        self.time_var = tk.DoubleVar()
        self.slider = tk.Scale(self.control_frame, variable=self.time_var, 
                              from_=0, to=self.total_frames, 
                              orient=tk.HORIZONTAL, showvalue=False,
                              command=self.slider_changed)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Time label
        self.time_label = tk.Label(self.control_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.RIGHT, padx=5)
        
        # Add measurements display frame
        self.measurements_frame = tk.Frame(parent_widget)
        self.measurements_frame.pack(fill=tk.X, expand=False, pady=5)
        
        # Create labels for measurements
        self.left_eye_title = tk.Label(self.measurements_frame, text="Left Eye", font=("Arial", 12, "bold"))
        self.left_eye_title.grid(row=0, column=0, padx=10, pady=5)
        
        self.left_vph_label = tk.Label(self.measurements_frame, text="VPH: --")
        self.left_vph_label.grid(row=1, column=0, padx=10, pady=2, sticky=tk.W)
        
        self.left_mrd1_label = tk.Label(self.measurements_frame, text="MRD1: --")
        self.left_mrd1_label.grid(row=2, column=0, padx=10, pady=2, sticky=tk.W)
        
        self.right_eye_title = tk.Label(self.measurements_frame, text="Right Eye", font=("Arial", 12, "bold"))
        self.right_eye_title.grid(row=0, column=1, padx=10, pady=5)
        
        self.right_vph_label = tk.Label(self.measurements_frame, text="VPH: --")
        self.right_vph_label.grid(row=1, column=1, padx=10, pady=2, sticky=tk.W)
        
        self.right_mrd1_label = tk.Label(self.measurements_frame, text="MRD1: --")
        self.right_mrd1_label.grid(row=2, column=1, padx=10, pady=2, sticky=tk.W)
        
        # State variables
        self.is_playing = False
        self.thread = None
        self.photo = None
        self.frame_skip = 1  # Skip frames for high FPS videos
        
        # For high frame rate videos, calculate frame skip
        if self.fps > 60:
            self.frame_skip = max(1, int(self.fps / 60))
            print(f"High frame rate video detected. Frame skip set to {self.frame_skip}")
        
        # Bind window close event
        self.parent.bind("<Destroy>", self.on_destroy)
        
        # Display first frame
        self.update_frame()
        
    def set_playback_speed(self, speed):
        """Change the playback speed"""
        self.playback_speed = speed
        self.display_fps = min(self.fps, 60.0) * self.playback_speed
        print(f"Playback speed set to {speed}x (Display FPS: {self.display_fps:.2f})")
        
        # If currently playing, restart with new speed
        if self.is_playing:
            was_playing = True
            self.pause()
        else:
            was_playing = False
            
        if was_playing:
            self.play()
        
    def connect_to_video_source(self):
        """Connect to video source, handling different input types"""
        if isinstance(self.video_source, int):
            # Camera source
            self.cap = cv2.VideoCapture(self.video_source)
        elif isinstance(self.video_source, str) and os.path.exists(self.video_source):
            # File source
            self.cap = cv2.VideoCapture(self.video_source)
        else:
            print(f"Invalid video source: {self.video_source}")
            self.cap = None
    
    def get_frame(self):
        """Get a frame from the video source"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return (ret, frame)
        return (False, None)
    
    def load_measurements_data(self, csv_file):
        """Load eye measurements data from CSV file"""
        try:
            self.measurements_data = pd.read_csv(csv_file)
            print(f"Loaded measurements data with {len(self.measurements_data)} entries")
        except Exception as e:
            print(f"Error loading measurements data: {str(e)}")
            self.measurements_data = None
            
    def get_frame_measurements(self, frame_number):
        """Get measurements for the current frame"""
        if self.measurements_data is None:
            return None
        
        try:
            # Find the row corresponding to this frame (or closest match)
            if 'timestamp' in self.measurements_data.columns:
                # If using timestamp, convert frame number to timestamp
                timestamp = frame_number / self.fps if self.fps > 0 else 0
                # Find closest timestamp
                closest_idx = (self.measurements_data['timestamp'] - timestamp).abs().idxmin()
                return self.measurements_data.iloc[closest_idx]
            elif frame_number < len(self.measurements_data):
                # If measurements are per frame, simply use the frame index
                return self.measurements_data.iloc[frame_number]
        except Exception as e:
            print(f"Error retrieving measurements for frame {frame_number}: {str(e)}")
        
        return None
    
    def update_measurements_display(self, frame_number):
        """Update the measurements display for the current frame"""
        measurements = self.get_frame_measurements(frame_number)
        
        if measurements is not None:
            # Update left eye measurements
            if 'left_palpebral_height' in measurements:
                self.left_vph_label.config(text=f"VPH: {measurements['left_palpebral_height']:.2f} mm")
            
            if 'left_pupil_to_lower' in measurements:
                self.left_mrd1_label.config(text=f"MRD1: {measurements['left_pupil_to_lower']:.2f} mm")
            
            # Update right eye measurements
            if 'right_palpebral_height' in measurements:
                self.right_vph_label.config(text=f"VPH: {measurements['right_palpebral_height']:.2f} mm")
            
            if 'right_pupil_to_lower' in measurements:
                self.right_mrd1_label.config(text=f"MRD1: {measurements['right_pupil_to_lower']:.2f} mm")
    
    def update_frame(self):
        """Update the current frame display"""
        ret, frame = self.get_frame()
        if ret:
            # Resize frame to exactly fit canvas dimensions while preserving aspect ratio
            # This ensures the image fills the canvas completely
            frame = cv2.resize(frame, (self.width, self.height))
            
            # Convert frame to PhotoImage
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            
            # Clear previous content and update canvas with centered image
            self.canvas.delete("all")
            # Draw image centered in the canvas
            self.canvas.create_image(self.width//2, self.height//2, image=self.photo, anchor=tk.CENTER)
            
            # Update time label
            current_time = self.current_frame_num / self.fps if self.fps > 0 else 0
            total_time = self.total_frames / self.fps if self.fps > 0 else 0
            
            self.time_label.config(text=f"{self.format_time(current_time)} / {self.format_time(total_time)}")
            
            # Update slider without triggering event
            self.slider.set(self.current_frame_num)
            
            # Update measurements display
            self.update_measurements_display(self.current_frame_num)
            
            # Increment frame counter
            self.current_frame_num += 1
    
    def format_time(self, seconds):
        """Format seconds to MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def play(self):
        """Start playing the video"""
        if not self.is_playing:
            self.is_playing = True
            # If we reached the end, restart
            if self.current_frame_num >= self.total_frames:
                self.current_frame_num = 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Start playback thread
            self.thread = threading.Thread(target=self._play_video)
            self.thread.daemon = True
            self.thread.start()
    
    def _play_video(self):
        """Video playback thread function"""
        # Calculate frame delay based on display FPS and playback speed
        frame_delay = 1.0 / self.display_fps if self.display_fps > 0 else 0.033  # Default to ~30fps
        
        print(f"Starting playback with frame delay: {frame_delay:.4f}s ({self.display_fps:.1f} FPS)")
        
        skip_counter = 0
        
        while self.is_playing and self.current_frame_num < self.total_frames:
            start_time = time.time()
            
            # For high frame rate videos, skip frames to maintain smooth playback
            if self.frame_skip > 1 and skip_counter < self.frame_skip - 1:
                # Skip this frame (just read it and increment counter)
                ret, _ = self.get_frame()
                if not ret:
                    break
                self.current_frame_num += 1
                skip_counter += 1
                continue
            else:
                skip_counter = 0  # Reset counter
            
            # Update frame
            self.update_frame()
            
            # Calculate delay to maintain correct FPS
            elapsed = time.time() - start_time
            delay = max(0, frame_delay - elapsed)
            
            # If processing is taking too long (more than 2x the frame delay),
            # we might need to skip frames to catch up
            if elapsed > frame_delay * 2:
                # Skip a frame next time
                skip_counter = min(skip_counter + 1, self.frame_skip)
                delay = 0  # No delay, we're already behind
            
            time.sleep(delay)
    
    def pause(self):
        """Pause video playback"""
        self.is_playing = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def stop(self):
        """Stop video playback and return to beginning"""
        self.is_playing = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        # Reset to first frame
        self.current_frame_num = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.update_frame()
    
    def slider_changed(self, event=None):
        """Handle slider position change"""
        # Pause playback
        was_playing = self.is_playing
        self.pause()
        
        # Set to new position
        pos = int(self.time_var.get())
        self.current_frame_num = pos
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        
        # Update display
        self.update_frame()
        
        # Resume playback if it was playing before
        if was_playing:
            self.play()
    
    def on_destroy(self, event=None):
        """Clean up resources when window is closed"""
        self.pause()
        if self.cap:
            self.cap.release()
    
    def open_video(self, video_source):
        """Open a new video source"""
        # Clean up existing resources
        self.pause()
        if self.cap:
            self.cap.release()
        
        # Set new source
        self.video_source = video_source
        self.connect_to_video_source()
        
        if self.cap is None or not self.cap.isOpened():
            print(f"Error: Unable to open video source {video_source}")
            return False
        
        # Get actual video dimensions
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate aspect ratio
        aspect_ratio = actual_width / actual_height
        
        # Get parent widget dimensions for scaling
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Use available space in parent widget
        available_width = int(parent_width * 0.8) if parent_width > 100 else 640
        available_height = int(parent_height * 0.6) if parent_height > 100 else 480
        
        # Preserve aspect ratio while maximizing space
        if available_width / available_height > aspect_ratio:
            # Available space is wider than video
            self.height = available_height
            self.width = int(available_height * aspect_ratio)
        else:
            # Available space is taller than video
            self.width = available_width
            self.height = int(available_width / aspect_ratio)
        
        # Update other properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Cap the FPS at 60 for display purposes if it's very high (slow motion videos)
        self.display_fps = min(self.fps, 60.0) * self.playback_speed
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_num = 0
        
        # For high frame rate videos, calculate frame skip
        if self.fps > 60:
            self.frame_skip = max(1, int(self.fps / 60))
            print(f"High frame rate video detected. Frame skip set to {self.frame_skip}")
        else:
            self.frame_skip = 1
        
        # Update frame and canvas size
        self.video_frame.config(width=self.width, height=self.height)
        self.canvas.config(width=self.width, height=self.height)
        
        # Update slider
        self.slider.config(to=self.total_frames)
        
        # Display first frame
        self.update_frame()
        return True 