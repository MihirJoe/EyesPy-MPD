#!/usr/bin/env python3
"""
Extract and Measure Eyes with SAM-based Iris Detection

This script:
1. Takes a video file as input
2. Uses EyeFrameExtractor to extract left and right eye frames
3. Uses SAM (Segment Anything Model) to detect iris/cornea
4. Normalizes measurements to 11.8mm iris diameter
5. Saves them to a folder
6. Optionally displays the results using the existing CustomVideoPlayer

Usage:
  python extract_and_measure_eyes.py <video_file> [--no-display] [--iris-diameter DIAMETER] [--sam-checkpoint PATH]
"""

import os
import sys
import argparse
from eye_frame_extractor_calibrated import EyeFrameExtractor
import tkinter as tk
from custom_video_player import CustomVideoPlayer

def extract_and_measure(video_path, display_results=True, iris_diameter=11.8, verbose=True, 
                        scaling_factor=3.0, pupil_correction=0.4, sam_checkpoint="sam_vit_h_4b8939.pth", 
                        visualize_frame=None):
    """
    Extract eyes from video, measure them using SAM for iris detection, and optionally display results
    
    Args:
        video_path (str): Path to the video file
        display_results (bool): Whether to display the results
        iris_diameter (float): Known iris diameter in millimeters for calibration (default 11.8mm)
        verbose (bool): Whether to print measurements by frame in the console
        scaling_factor (float): Additional scaling factor to adjust measurements
        pupil_correction (float): Correction factor for pupil diameter calculation
        sam_checkpoint (str): Path to the SAM model checkpoint file
        visualize_frame (int): Frame number to visualize iris detection using SAM model
        
    Returns:
        tuple: (eye_data_filename, eye_images_folder)
    """
    print(f"Processing video: {video_path}")
    print(f"Using iris diameter of {iris_diameter} mm for calibration")
    print(f"Using scaling factor of {scaling_factor} and pupil correction of {pupil_correction}")
    
    if sam_checkpoint and os.path.exists(sam_checkpoint):
        print(f"Using SAM model for iris detection with checkpoint: {sam_checkpoint}")
    else:
        if sam_checkpoint:
            print(f"Warning: SAM checkpoint '{sam_checkpoint}' not found")
        print("Falling back to traditional detection method")
        sam_checkpoint = None
    
    if visualize_frame is not None:
        print(f"Will visualize iris detection for frame: {visualize_frame}")
    
    if verbose:
        print("Verbose mode enabled - measurements will be printed for each frame")
    
    # Extract eye frames and measurements using the calibrated extractor
    extractor = EyeFrameExtractor(video_path, pupil_diameter_mm=iris_diameter, 
                                 verbose=verbose, scaling_factor=scaling_factor,
                                 pupil_correction=pupil_correction,
                                 sam_checkpoint=sam_checkpoint)
    
    # Set the frame to visualize
    if visualize_frame is not None:
        extractor.visualize_frame = visualize_frame
    
    csv_file = extractor.extract_eye_frames()
    
    if not csv_file:
        print("Error: Failed to extract eye measurements")
        return None, None
    
    print(f"Eye measurement data saved to: {csv_file}")
    print(f"Eye frames saved to: {extractor.output_dir}")
    
    if display_results:
        display_results_with_player(video_path, csv_file, extractor.output_dir)
    
    return csv_file, extractor.output_dir

def display_results_with_player(video_path, eye_data_filename, eye_images_folder):
    """
    Display the results using the CustomVideoPlayer
    
    Args:
        video_path (str): Path to the video file
        eye_data_filename (str): Path to the CSV file with eye measurements
        eye_images_folder (str): Path to the folder with eye images
    """
    # Create a root window
    root = tk.Tk()
    root.title("EyesPy - Eye Measurements Viewer")
    
    # Get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Set window size (80% of screen)
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
    
    # Center the window
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    
    # Create a frame for the video player
    video_frame = tk.Frame(root)
    video_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Create a header
    header_label = tk.Label(root, text="EyesPy - Eye Tracking Measurements", font=("Arial", 16, "bold"))
    header_label.pack(pady=10)
    
    # Create info label
    info_label = tk.Label(root, text=f"Video: {os.path.basename(video_path)}", font=("Arial", 12))
    info_label.pack(pady=5)
    
    # Initialize video player with measurements
    try:
        video_player = CustomVideoPlayer(video_frame, video_path, 
                                     width=int(window_width*0.8), 
                                     height=int(window_height*0.6))
        
        # Load measurements data
        video_player.load_measurements_data(eye_data_filename)
        
        # Create controls frame
        controls_frame = tk.Frame(root)
        controls_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Add exit button
        exit_button = tk.Button(controls_frame, text="Exit", command=root.destroy)
        exit_button.pack(side=tk.RIGHT, padx=10)
        
        # Start playing the video
        video_player.play()
        
        # Run the Tkinter event loop
        root.mainloop()
        
    except Exception as e:
        print(f"Error displaying video: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract eyes from video and measure using SAM for iris detection")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--no-display", action="store_true", help="Don't display results")
    parser.add_argument("--iris-diameter", type=float, default=11.8, 
                        help="Known iris diameter in millimeters for calibration (default: 11.8mm)")
    parser.add_argument("--verbose", action="store_true", 
                        help="Print detailed eye measurements for each frame as they are processed")
    parser.add_argument("--scaling-factor", type=float, default=3.0,
                        help="Scaling factor to adjust measurement calibration (default: 3.0)")
    parser.add_argument("--pupil-correction", type=float, default=0.4,
                        help="Correction factor for pupil diameter calculation (default: 0.4)")
    parser.add_argument("--sam-checkpoint", default="sam_vit_h_4b8939.pth",
                        help="Path to the SAM model checkpoint file (default: sam_vit_h_4b8939.pth)")
    parser.add_argument("--visualize-frame", type=int,
                        help="Visualize a specific frame's iris detection using SAM model")
    
    args = parser.parse_args()
    
    # Extract and measure eyes
    eye_data_filename, eye_images_folder = extract_and_measure(
        args.video_path,
        display_results=not args.no_display,
        iris_diameter=args.iris_diameter,
        verbose=args.verbose,
        scaling_factor=args.scaling_factor,
        pupil_correction=args.pupil_correction,
        sam_checkpoint=args.sam_checkpoint,
        visualize_frame=args.visualize_frame
    )
    
    if eye_data_filename:
        print("\nProcessing complete!")
        print(f"Eye measurement data: {eye_data_filename}")
        print(f"Eye frames folder: {eye_images_folder}")
    else:
        print("Processing failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 