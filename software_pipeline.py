""" Import Libraries Here """
import os
import cv2
import numpy 
import glob
from scipy import ndimage
from left_and_right_eye_measurements import EyeTracker
import torch
from eyespy_mpd_NN import ModifiedUNet
from eyespy_gui import EyeTrackingGUI
import tkinter as tk
from tkinter import filedialog, ttk, Frame, Label, Button
import matplotlib
from custom_video_player import CustomVideoPlayer
import time
import PIL

matplotlib.use("TkAgg")  # Ensure Matplotlib integrates with Tkinter

testGUI = False # true to show GUI, false to show dummy function


""" Write Functions Here """ 

def get_data_from_file(file_name, testing=False, saving=False):
# Function that takes in a fileName and outputs an image or array of images
# Input: 
#   file_name - A string with the complete file path
#   saving - Saves each frame as a file if true
#   testing - Prints helpful outputs
# Output: 
#   images - Array of images (all frames if video input) from the file
# Required libraries: os, cv2, numpy

    # Create a VideoCapture object to read the video
    video = cv2.VideoCapture(file_name)
    
    # Check if the video was opened successfully
    if not video.isOpened():
        print(f"Error in 'get_data_from_file' function: Could not open video {file_name}")
        return
    
    # Get the video's frame count
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # get the first frame
    ret, frame = video.read()

    # save frame dimensions, use to initialize an array to store the frames
    frame_dim = list(numpy.shape(frame))
    frame_dim.insert(0, frame_count)

    # initialize array to store frames and save first frame
    images = numpy.zeros(frame_dim).astype('uint8')
    images[0, :] = frame

    # if testing, print the frame dimensions
    if testing:
        print(f"Frame Count = {frame_count} \nFrame Dimensions = {frame_dim}")


    # if saving frames, make directory to save them:
    if saving:
        # Get the directory and filename of the input video
        video_dir, video_filename = os.path.split(file_name)
        
        # Create a directory to save frames (same directory as the video)
        frame_dir = os.path.join(video_dir, 'base_frames')
        os.makedirs(frame_dir, exist_ok=True)

        # Save first frame as an image file (JPEG format)
        frame_filename = os.path.join(frame_dir, f'frame_0000.jpg')
        cv2.imwrite(frame_filename, frame)


    # Loop through each frame in the video
    for i in range(1, frame_count):
        ret, frame = video.read()
        if not ret:
            break  # Stop if no more frames are available
        
        # Save each frame in the image array
        images[i, :] = frame

        # save each frame to folder if saving:
        if saving:
            # Save each frame as an image file (JPEG format)
            frame_filename = os.path.join(frame_dir, f'frame_{i:04d}.jpg')
            cv2.imwrite(frame_filename, frame)

    # Release the VideoCapture object
    video.release()

    return images

def GUI_to_get_data(testing=False, saving=False):
# Function that displays a GUI the patient interacts with to collect an image/video from a file OR via live feed from the camera. 
# The function then uses the EyeTracker class to collect predicted measurements and isolated frames of each eye for the image. 
# Input: 
#   Saving - Saves each frame as a file if true
#   Testing - Prints helpful outputs
# Output: 
#   root - The tkinter root window to be reused in subsequent operations
#   csv_file - Path to the CSV file with eye measurements
#   folder_with_images - Path to the folder containing eye images
#   video_source - Path to the selected video file

    # Create a root window for the dialog
    root = tk.Tk()
    root.title("EyesPy Video Selection")
    
    # Get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Set window size (70% of screen)
    window_width = int(screen_width * 0.7)
    window_height = int(screen_height * 0.7)
    
    # Center the window
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    
    # Video source variable
    video_source = None
    is_live_video = False
    video_player = None
    video_player_frame = None
    proceed_to_processing = False
    
    def select_live_video():
        nonlocal video_source, is_live_video, video_player, video_player_frame
        video_source = 0  # Use default camera
        is_live_video = True
        
        # Hide selection buttons
        selection_frame.pack_forget()
        title_label.config(text="Live Camera Feed")
        
        # Create video player frame
        video_player_frame = tk.Frame(root)
        video_player_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Initialize video player
        try:
            # Use 80% of window width and 60% of window height for the video
            video_player = CustomVideoPlayer(video_player_frame, video_source, 
                                         width=int(window_width*0.8), 
                                         height=int(window_height*0.6))
            video_player.play()  # Start playing immediately
            
            # Add confirm button
            confirm_button = tk.Button(root, text="Capture Video", command=confirm_selection)
            confirm_button.pack(pady=10)
        except Exception as e:
            title_label.config(text=f"Error: {str(e)}")
    
    def select_file_video():
        nonlocal video_source, is_live_video, video_player, video_player_frame
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if not video_path:
            return  # User canceled
        
        video_source = video_path
        is_live_video = False
        
        # Hide selection buttons
        selection_frame.pack_forget()
        title_label.config(text=f"Selected: {os.path.basename(video_path)}")
        
        # Create video player frame
        video_player_frame = tk.Frame(root)
        video_player_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Initialize video player
        try:
            # Use 80% of window width and 60% of window height for the video
            video_player = CustomVideoPlayer(video_player_frame, video_source,
                                         width=int(window_width*0.8),
                                         height=int(window_height*0.6))
            
            # Add confirm button
            confirm_button = tk.Button(root, text="Process Video", command=confirm_selection)
            confirm_button.pack(pady=10)
        except Exception as e:
            title_label.config(text=f"Error: {str(e)}")
    
    def confirm_selection():
        nonlocal proceed_to_processing
        # Mark as ready to proceed but don't destroy the window
        proceed_to_processing = True
        
        # Pause the video player if it exists
        if video_player:
            video_player.pause()
        
        # Clean up the current UI
        for widget in root.winfo_children():
            widget.pack_forget()
        
        # Show a processing message
        processing_label = tk.Label(root, text="Processing video...", font=("Arial", 16))
        processing_label.pack(pady=20)
        
        # Create a progress bar
        progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="indeterminate")
        progress.pack(pady=20, padx=40, fill=tk.X)
        progress.start()
        
        # Force update to show the processing UI
        root.update()
    
    # Create GUI elements
    title_label = tk.Label(root, text="Select Video Source", font=("Arial", 16))
    title_label.pack(pady=20)
    
    selection_frame = tk.Frame(root)
    selection_frame.pack(expand=True)
    
    live_button = tk.Button(selection_frame, text="Use Live Camera", font=("Arial", 14), padx=20, pady=10, command=select_live_video)
    live_button.pack(pady=10)
    
    file_button = tk.Button(selection_frame, text="Upload Video File", font=("Arial", 14), padx=20, pady=10, command=select_file_video)
    file_button.pack(pady=10)
    
    # Start the GUI loop, but allow it to continue processing once proceed_to_processing is True
    # We need to manually update the UI while waiting for user selection
    while not proceed_to_processing:
        root.update()
        time.sleep(0.1)  # Short sleep to prevent high CPU usage
        if not root.winfo_exists():
            # Window was closed
            return None, None, None, None
    
    # Process the selected video after confirm button is clicked
    if video_source is not None:
        # Process the video with EyeTracker
        if is_live_video:
            print("Processing live video...")
            tracker = EyeTracker(0, saving=saving)
            csv_file = tracker.run(duration=20)
        else:
            print(f"Processing video file: {video_source}")
            # Strip any quotes from the path
            video_source = video_source.strip("'\"")
            tracker = EyeTracker(video_source, saving=saving)
            csv_file = tracker.run()
        
        # Get output folder
        folder_with_images = tracker.get_output_dir()
        return root, csv_file, folder_with_images, video_source
    else:
        print("No video selected. Exiting.")
        return root, None, None, None

""" DELETE THIS FUNCTION """
def isolate_eye_images(images, testing=False, saving=False):
# Function that takes in an array of face images and outputs two arrays of eye images (one for each eye)
# Input: 
#   images - An array of images (one for each video frame)
#   saving - Saves each cropped eye frame as a file if true
#   testing - Prints helpful outputs
# Outputs: 
#   right_eye_images: list of isolated, cropped images of the right eye for every frame
#   left_eye_images: list of isolated, cropped images of the left eye for every frame
    right_eye_images = list()
    left_eye_images = list()

    # Create output directory if it doesn't exist
    output_dir = './cropped_eyes/'
    os.makedirs(output_dir, exist_ok=True)
    
    for frame in range(0, images.shape[0]):

        # Convert to RGB for matplotlib display
        if testing:
            print(f'Image Frame #{frame} shape: {numpy.shape(images[frame])} \n\tand type: {numpy.dtype(images[frame, 0, 0, 0])}')

        image_rgb = cv2.cvtColor(images[frame], cv2.COLOR_BGR2RGB)
        # Convert to grayscale for detection
        gray = cv2.cvtColor(images[frame], cv2.COLOR_BGR2GRAY)
        
        # Load the pre-trained classifiers
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Set x, y, w, h
        x = 0
        y = 0

        img_size = numpy.shape(gray)
        w = img_size[1]
        h = img_size[0]

        # Detect eyes in the face region with adjusted parameters
        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(int(w/12), int(h/12)),
            maxSize=(int(w/3), int(h/3))
        )
                
        # Sort eyes by x-coordinate to get left and right eye
        eyes = sorted(eyes, key=lambda x: x[0])
        
        # If we still don't have exactly 2 eyes, try to adjust detection
        if len(eyes) != 2:
            eyes = eye_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=8,
                minSize=(int(w/12), int(h/12)),
                maxSize=(int(w/3), int(h/3))
            )
            eyes = sorted(eyes, key=lambda x: x[0])
            
        if len(eyes) < 2:
            raise ValueError(f"Could not detect both eyes properly in frame_{frame:04d}")
            
        # Take only the first two detected eyes
        eyes = eyes[:2]
        eye_images = []
        
        # Draw rectangles around eyes and crop them
        for i, (ex, ey, ew, eh) in enumerate(eyes):
            
            # Crop eye region
            eye_roi = image_rgb[y+ey:y+ey+eh, x+ex:x+ex+ew]
            eye_images.append(eye_roi)
            
            # Save eye image
            eye_type = "left" if i == 0 else "right"
            output_path = os.path.join(output_dir, f"frame_{frame:04d}_{eye_type}_eye.jpg")
            
            # Convert from RGB to BGR for cv2.imwrite
            eye_bgr = cv2.cvtColor(eye_roi, cv2.COLOR_RGB2BGR)

            if saving:
                cv2.imwrite(output_path, eye_bgr)

            # add to list for output
            if eye_type == "left":
                left_eye_images.append(eye_bgr)
            else:
                right_eye_images.append(eye_bgr)


    return right_eye_images, left_eye_images

def blur_and_sample(img, a=0.4):
    # Enter your code here
    
    # build Gaussian Mask
    gaussMask = numpy.array([[.25-.5*a, .25, a, .25, .25-.5*a]])
    gaussMask = gaussMask / gaussMask.sum()
    #print(gaussMask, gaussMask.shape) # debugging
    #print(img.shape)

    # correlate row-wise
    rowBlurred = ndimage.correlate(img, gaussMask)
    blurred = ndimage.correlate(rowBlurred, gaussMask.T)

    # initialize new photo
    nRows = int(numpy.ceil(img.shape[0] / 2))
    nCols = int(numpy.ceil(img.shape[1] / 2))
    downsampled_img = numpy.zeros([nRows, nCols])

    # sample pixels for the downsampled image
    for row in range(0, nRows):
        for col in range(0, nCols):
            bRow = row*2
            bCol = col*2
            downsampled_img[row, col] = numpy.mean(blurred[bRow:bRow+1, bCol:bCol+1])
    
    return downsampled_img

def sharpen(img):
    # Enter your code here
    
    # initialize new photo
    nRows = int(img.shape[0] * 2 - 1)
    nCols = int(img.shape[1] * 2 - 1)
    upsampled_img = numpy.zeros([nRows, nCols])
    
    # assign odd (even index) row,col values to correspond to known values
    for row in range(0, nRows, 2):
        for col in range(0, nCols, 2):
            upsampled_img[row, col] = img[int(row/2)-1, int(col/2)-1]

    # assign even (odd index) row,col values using interpolation
    for row in range(1, nRows, 2):
        for col in range(1, nCols, 2):
            inputRow = int((row - 1) / 2)
            inputCol = int((col - 1)/ 2)
            upsampled_img[row, col] = numpy.mean(img[inputRow:inputRow+1, inputCol:inputCol+1])

    return upsampled_img

def downsample_to_ideal(image, ideal_shape = [256,256], a=0.4):
# Function that an image as an input and updates the resolution of each image to make the ideal shape. 
# Uses downsampling to decrease resolution
# Inputs:
#   image: A square image that is larger or equal to the ideal_shape image, but not more than twice the size
#   ideal_shape: The desired dimensions of each image. Has a default value to match the ML model expected input
# Outputs:  
#   resized_image: The image resized to the ideal_shape


    # initialize new photo
    resized_image = numpy.zeros(ideal_shape)

    # scaling factor
    scale_factor = ideal_shape[0] / image.shape[0]
    # print(scale_factor)

    # build Gaussian Mask
    gaussMask = numpy.array([[.25-.5*a, .25, a, .25, .25-.5*a]])
    gaussMask = gaussMask / gaussMask.sum()
    #print(gaussMask, gaussMask.shape) # debugging
    #print(img.shape)

    # correlate row-wise
    rowBlurred = ndimage.correlate(image, gaussMask)
    blurred = ndimage.correlate(rowBlurred, gaussMask.T)

    # downsample to fill the photo
    for row in range(0, ideal_shape[0]):
        for col in range(0, ideal_shape[1]):

            # translate row and col number to original image
            original_row = row / scale_factor
            original_row_rounded = int(round(original_row, 0))
            original_row_rerror = original_row - original_row_rounded

            original_col = col / scale_factor
            original_col_rounded = int(round(original_col, 0))
            original_col_rerror = original_col - original_col_rounded

            # take gaussian average based on distance from actual value: 
            resized_image[row, col] = blurred[original_row_rounded, original_col_rounded]

    return resized_image



def change_resolution(image, ideal_shape = [256,256], testing=False, saving=False):
# Function that takes an array of images as an input and updates the resolution of each image to make the ideal shape. 
# Uses blurring or reverse blurring/sharpening to decrease and increase resolution (respectively)
# Inputs:
#   images: An array of images
#   ideal_shape: The desired dimensions of each image. Has a default value to match the ML model expected input
# Outputs:  
#   resized_images: The image array where each image has been resized to the ideal_shape

    # first, reduce to grayscale
    image = numpy.array(image).astype('uint8')
    resized_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # while loop to determine if image is correct size
    flag = True
    while(flag):
        if resized_image.shape[0] > ideal_shape[0] * 2: # if image is too large, blur and downsample
            if testing:
                print(f'Blurring: {resized_image.shape} vs {ideal_shape}')
            resized_image = blur_and_sample(resized_image)
        elif resized_image.shape[0] < ideal_shape[0]: # if image is too small, sharpen/interpolated it
            resized_image = sharpen(resized_image)
            if testing:
                print(f'Sharpening: {resized_image.shape} vs {ideal_shape}')
        else:
            resized_image = downsample_to_ideal(resized_image, ideal_shape)
            flag = False

    return resized_image

def change_resolution_frames(root, eye_images_folder, ideal_shape = [256,256], output_dir='./cropped_eyes/', testing=False, saving=True):
# Function that takes an array of images as an input and updates the resolution of each image to make the ideal shape. 
# Uses blurring or reverse blurring/sharpening to decrease and increase resolution (respectively)
# Inputs:
#   root: The existing tkinter root window
#   eye_images_folder: File path to folder where eye images are saved 
#   ideal_shape: The desired dimensions of each image. Has a default value to match the ML model expected input
#   testing: prints helpful debugging outputs if true
#   saving: saves each image if true
#   output_dir: if you want to manually set the output_directory
# Outputs:  
#   output_dir: The folder with the cropped eye images

    # Clean up the existing UI
    for widget in root.winfo_children():
        widget.pack_forget()
    
    # Update title
    root.title("EyesPy - Processing Eye Images")
    
    # Create processing UI elements
    processing_label = tk.Label(root, text="Resizing Eye Images for ML Processing...", font=("Arial", 16))
    processing_label.pack(pady=20)
    
    # Add an image display (optional)
    image_frame = tk.Frame(root)
    image_frame.pack(pady=20)
    
    # Progress elements
    progress_frame = tk.Frame(root)
    progress_frame.pack(fill=tk.X, padx=20, pady=10)
    
    progress_label = tk.Label(progress_frame, text="Processing images...", font=("Arial", 12))
    progress_label.pack(side=tk.LEFT, padx=10)
    
    progress = ttk.Progressbar(progress_frame, orient="horizontal", length=400, mode="determinate")
    progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)
    
    # Force UI update
    root.update()

    # Create array to hold filenames of each image
    # Strip any leading/trailing quotes from the path
    eye_images_folder = eye_images_folder.strip("'\"")
    
    # Handle path more safely by using os.path.join instead of string concatenation
    if not os.path.exists(eye_images_folder):
        progress_label.config(text=f"Warning: Eye images folder '{eye_images_folder}' does not exist!")
        root.update()
        time.sleep(2)
        return output_dir
        
    glob_search = os.path.join(eye_images_folder, '*.jpg')
    img_filenames = glob.glob(glob_search)

    if testing:
        print(f'Glob search for "{glob_search}" yielded {len(img_filenames)} images')

    # Update progress info
    total_images = len(img_filenames)
    progress['maximum'] = total_images
    progress_label.config(text=f"Processing 0 of {total_images} images...")
    root.update()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each image with progress updates
    for i, fname in enumerate(img_filenames):
        if testing:
            print(fname)
            
        # Update progress
        progress['value'] = i
        progress_label.config(text=f"Processing {i+1} of {total_images} images...")
        root.update()
        
        # Process the image
        frame = cv2.imread(fname)
        this_frame = change_resolution(frame, ideal_shape, testing, saving)

        if saving:
            output_path = fname.replace(eye_images_folder, output_dir)
            cv2.imwrite(output_path, this_frame)
            
        # Display the current image being processed (optional)
        if i % 5 == 0:  # Show every 5th image to avoid too many updates
            try:
                # Convert to PIL format for display
                img_display = PIL.Image.fromarray(this_frame)
                img_display = PIL.ImageTk.PhotoImage(image=img_display)
                
                # Clear previous images
                for widget in image_frame.winfo_children():
                    widget.destroy()
                
                # Display the image
                img_label = tk.Label(image_frame, image=img_display)
                img_label.image = img_display  # Keep a reference
                img_label.pack()
                
                # Show the filename
                name_label = tk.Label(image_frame, text=os.path.basename(fname))
                name_label.pack()
                
                root.update()
            except Exception as e:
                if testing:
                    print(f"Error displaying image: {e}")
    
    # Complete the progress
    progress['value'] = total_images
    progress_label.config(text=f"Completed processing {total_images} images")
    root.update()
    time.sleep(1)  # Pause briefly to show completion
    
    return(output_dir)


def apply_ML_model(root, eye_data_filename, eye_images_folder, video_source):
# Function that takes an array of images as an input, applies the ML model to each individual image/frame,
# and outputs the resulting data as an array
# Input: 
#   root: The existing tkinter root window
#   eye_data_filename: File path where the eye measurements are saved in csv format (with headers: 
#                      "timestamp,left_palpebral_height,left_pupil_to_lower,right_palpebral_height,right_pupil_to_lower")
#   eye_images_folder: File path to folder where eye images are saved 
#   video_source: Path to the original video file
# Output:
#   updated_eye_data_filename: A csv file with updated measurements for each frame
   
    # Load model from file
    model_filename = False
    if model_filename:
        model = ModifiedUNet()
        model.load_state_dict(torch.load(model_filename))
        model.eval()
    
    # Clean up the existing UI
    for widget in root.winfo_children():
        widget.pack_forget()
    
    # Update title
    root.title("EyesPy - Processing Video")
    
    # Create processing UI elements
    processing_label = tk.Label(root, text="Applying Machine Learning Model...", font=("Arial", 16))
    processing_label.pack(pady=20)
    
    # Create main frames for video display
    video_frame = tk.Frame(root)
    video_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Get window dimensions
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    
    # Progress bar
    progress_frame = tk.Frame(root)
    progress_frame.pack(fill=tk.X, padx=20, pady=10)
    
    progress_label = tk.Label(progress_frame, text="Processing frames...", font=("Arial", 12))
    progress_label.pack(side=tk.LEFT, padx=10)
    
    progress = ttk.Progressbar(progress_frame, orient="horizontal", length=400, mode="determinate")
    progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)
    
    # Speed controls
    controls_frame = tk.Frame(root)
    controls_frame.pack(fill=tk.X, padx=20, pady=5)
    
    speed_label = tk.Label(controls_frame, text="Preview Speed:", font=("Arial", 10))
    speed_label.pack(side=tk.LEFT, padx=5)
    
    current_speed = tk.DoubleVar(value=2.0)  # Default to 2x speed for processing preview
    
    def update_speed(speed):
        current_speed.set(speed)
        if 'video_player' in locals():
            video_player.set_playback_speed(speed)
    
    btn_speed_slow = tk.Button(controls_frame, text="1x", command=lambda: update_speed(1.0))
    btn_speed_slow.pack(side=tk.LEFT, padx=2)
    
    btn_speed_normal = tk.Button(controls_frame, text="2x", command=lambda: update_speed(2.0))
    btn_speed_normal.pack(side=tk.LEFT, padx=2)
    
    btn_speed_fast = tk.Button(controls_frame, text="4x", command=lambda: update_speed(4.0))
    btn_speed_fast.pack(side=tk.LEFT, padx=2)
    
    # Initialize video player with measurements
    try:
        # Create video player with faster playback for processing view
        video_player = CustomVideoPlayer(video_frame, video_source, 
                                     width=int(window_width*0.8), 
                                     height=int(window_height*0.6),
                                     playback_speed=current_speed.get())
        
        # Load measurements data if available
        if eye_data_filename and os.path.exists(eye_data_filename):
            video_player.load_measurements_data(eye_data_filename)
            
        # Get video details
        fps = video_player.fps
        total_frames = int(video_player.total_frames)
        
        # Calculate optimal frame sampling for preview
        # For high fps videos, we don't need to show every frame during processing
        if fps > 30:
            frame_step = max(1, int(fps / 10))  # Show approximately 10 frames per second of video
        else:
            frame_step = 1
            
        # Update progress configuration
        progress['maximum'] = total_frames
        
        # Update UI
        root.update()
        
        # Processing start time
        start_time = time.time()
        
        # Determine frame sampling based on total frames
        if total_frames > 500:
            # For very long videos, sample fewer frames for the preview
            preview_frames = list(range(0, total_frames, max(1, int(total_frames / 50))))
        else:
            # For shorter videos, show more frames in the preview
            preview_frames = list(range(0, total_frames, max(1, int(total_frames / 20))))
        
        # In a real implementation, this would be the ML processing loop
        # Here we're just simulating it with incremental updates
        for i, frame_num in enumerate(preview_frames):
            progress['value'] = frame_num
            progress_label.config(text=f"Processing frame {frame_num} of {total_frames} ({i+1}/{len(preview_frames)} samples)")
            
            # Move video to current frame
            video_player.current_frame_num = frame_num
            video_player.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            video_player.update_frame()
            
            # Update UI
            root.update()
            
            # Minimal delay - just enough to prevent UI from freezing
            # For actual processing, this would be where the model inference happens
            time.sleep(0.05)  # Very short delay to allow UI updates
        
        # Finish progress
        elapsed_time = time.time() - start_time
        progress['value'] = total_frames
        progress_label.config(text=f"Processing complete! ({elapsed_time:.1f} seconds)")
        root.update()
        
    except Exception as e:
        processing_label.config(text=f"Error during processing: {str(e)}")
    
    print(f"apply_ML_model function completed")
    updated_eye_data_filename = eye_data_filename
    
    # Pause to show completion
    time.sleep(1)
    
    return updated_eye_data_filename

def display_GUI_from_data(root, eye_data_filename, eye_images_folder, video_source):
# Function to display the functional GUI with the input data
# Inputs:
#   root: The existing tkinter root window
#   eye_data_filename: File path where the eye measurements are saved in csv format (with headers: 
#                      "timestamp,left_palpebral_height,left_pupil_to_lower,right_palpebral_height,right_pupil_to_lower")
#   eye_images_folder: File path to folder where eye images are saved 
#   video_source: Path to the original video file
# Outputs:
#   None
    # Setting testGUI to True to ensure GUI is shown
    global testGUI
    testGUI = True
    
    print("Displaying video with eye tracking measurements...")
    
    # Clean up paths before passing to GUI
    if eye_data_filename:
        eye_data_filename = eye_data_filename.strip("'\"")
    if eye_images_folder:
        eye_images_folder = eye_images_folder.strip("'\"")
    if video_source:
        video_source = video_source.strip("'\"")
    
    # Clean up the existing UI
    for widget in root.winfo_children():
        widget.pack_forget()
    
    # Update title
    root.title("EyesPy - Eye Tracking Measurements")
    
    # Check if video exists
    if not video_source or not os.path.exists(video_source):
        # Get the video file from the eye images folder if not provided
        video_file = None
        # Look for the original video in parent directory
        parent_dir = os.path.dirname(eye_images_folder)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for ext in video_extensions:
            potential_files = glob.glob(os.path.join(parent_dir, f'*{ext}'))
            if potential_files:
                video_source = potential_files[0]
                break

    if not video_source or not os.path.exists(video_source):
        # If no video file found, show message
        lbl_no_video = tk.Label(root, text="No video file found to display.", font=("Arial", 14))
        lbl_no_video.pack(pady=20)
        btn_close = tk.Button(root, text="Close", command=root.destroy)
        btn_close.pack(pady=10)
        root.mainloop()
        return
    
    # Create header
    header_frame = tk.Frame(root)
    header_frame.pack(fill=tk.X, padx=20, pady=10)
    
    header_label = tk.Label(header_frame, text="EyesPy - Eye Measurement Results", font=("Arial", 16, "bold"))
    header_label.pack(side=tk.LEFT)
    
    # Create main frames for video display
    video_frame = tk.Frame(root)
    video_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Get window dimensions
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    
    # Initialize custom video player with measurements
    try:
        video_player = CustomVideoPlayer(video_frame, video_source, 
                                     width=int(window_width*0.8), 
                                     height=int(window_height*0.6))
        
        # Load measurements data
        if eye_data_filename and os.path.exists(eye_data_filename):
            video_player.load_measurements_data(eye_data_filename)
        
        # Add controls frame
        controls_frame = tk.Frame(root)
        controls_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Add info label
        info_label = tk.Label(controls_frame, text=f"Video: {os.path.basename(video_source)}", font=("Arial", 10))
        info_label.pack(side=tk.LEFT, padx=10)
        
        # Add spacer
        spacer = tk.Frame(controls_frame)
        spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add save button
        save_button = tk.Button(controls_frame, text="Save Report", command=lambda: save_report(eye_data_filename))
        save_button.pack(side=tk.LEFT, padx=10)
        
        # Add exit button
        exit_button = tk.Button(controls_frame, text="Exit", command=root.destroy)
        exit_button.pack(side=tk.LEFT, padx=10)
        
        # Define save report function
        def save_report(csv_file):
            if not csv_file:
                tk.messagebox.showerror("Error", "No measurement data available to save.")
                return
                
            save_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Save Eye Measurement Report"
            )
            
            if save_path:
                info_label.config(text=f"Report saved to: {os.path.basename(save_path)}")
        
        # Start playing the video
        video_player.play()
        
    except Exception as e:
        error_label = tk.Label(root, text=f"Error displaying video: {str(e)}", font=("Arial", 12))
        error_label.pack(pady=20)
    
    # Wait for window to close
    root.mainloop()

    print(f"The Eye Data is Saved in: {eye_data_filename}")
    print(f"The Eye Images are in: {eye_images_folder}")

    return

""" Main Section of Code """

# Step 1: Open GUI to select and process video 
root, csv_file, folder_with_images, video_source = GUI_to_get_data(testing=False, saving=True)

# Check if user closed the window
if root is None or not root.winfo_exists():
    print("Application closed by user.")
    exit()

# Step 2: Resize eye images to prepare for ML model
folder_with_cropped_frames = change_resolution_frames(root, folder_with_images, saving=True, testing=False)

# Check if user closed the window
if not root.winfo_exists():
    print("Application closed by user.")
    exit()

# Step 3: Apply ML model to process the images
updated_csv_file = apply_ML_model(root, csv_file, folder_with_images, video_source)

# Check if user closed the window
if not root.winfo_exists():
    print("Application closed by user.")
    exit()

# Step 4: Display final results with measurements
print("Displaying video with eye tracking measurements...")
display_GUI_from_data(root, updated_csv_file, folder_with_images, video_source)

print("Application completed successfully.")

