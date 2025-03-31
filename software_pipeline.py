
""" Import Libraries Here """
import os
import cv2
import numpy 
import glob
from scipy import ndimage
from left_and_right_eye_measurements import EyeTracker
import torch
from eyespy_mpd_NN import ModifiedUNet


testOnlyGUI = False # true to show only GUI, false to show normal operation
testGUI = True # true to show GUI, false to show dummy function


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
#   csv_file: file name for csv file with eye measurements for each frame for the each eye (VPH, MRD1)
#   folder_with_images: file name for where cropped, color left and right eye images are stored

    # for now - request file name from screen
    tracker = EyeTracker(saving=True, testing = False)
    tracker.run()

    # save output file names
    csv_file = tracker.csv_file
    folder_with_images = tracker.get_output_dir()


    return csv_file, folder_with_images

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

def change_resolution_frames(eye_images_folder, ideal_shape = [256,256], output_dir='./cropped_eyes/', testing=False, saving=True):
# Function that takes an array of images as an input and updates the resolution of each image to make the ideal shape. 
# Uses blurring or reverse blurring/sharpening to decrease and increase resolution (respectively)
# Inputs:
#   eye_images_folder: File path to folder where eye images are saved 
#   ideal_shape: The desired dimensions of each image. Has a default value to match the ML model expected input
#   testing: prints helpful debugging outputs if true
#   saving: saves each image if true
#   output_dir: if you want to manually set the output_directory
# Outputs:  
#   output_dir: The folder with the cropped eye images

    # create array to hold filenames of each image
    glob_search = './' + eye_images_folder + '/' + '*' + '.jpg'
    img_filenames = glob.glob(glob_search)

    if testing:
        print(f'Glob search for "{glob_search}" yielded {len(img_filenames)} images' )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for fname in img_filenames:
        if testing:
            print(fname)
        frame = cv2.imread(fname)
        this_frame = change_resolution(frame, ideal_shape, testing, saving)

        if saving:
            output_path = fname.replace(eye_images_folder, output_dir)
            cv2.imwrite(output_path, this_frame)
    
    return(output_dir)


def apply_ML_model(eye_data_filename, eye_images_folder):
# Function that takes an array of images as an input, applies the ML model to each individual image/frame,
# and outputs the resulting data as an array
# Input: 
#   eye_data_filename: File path where the eye measurements are saved in csv format (with headers: 
#                      "timestamp,left_palpebral_height,left_pupil_to_lower,right_palpebral_height,right_pupil_to_lower")
#   eye_images_folder: File path to folder where eye images are saved 
# Output:
#   updated_eye_data_filename: A csv file with updated measurements for each frame
#                        (UPDATE TO INLCUDE PARAMETERS IN WHICH COLUMNS)
   
    # load model from file
    model_filename = False
    if model_filename:
        model = ModifiedUNet()
        model.load_state_dict(torch.load(model_filename))
        model.eval()
    

    print(f"apply_ML_model function is currently blank")
    updated_eye_data_filename = eye_data_filename

    return updated_eye_data_filename

def display_GUI_from_data(eye_data_filename, eye_images_folder):
# Function to display the functional GUI with the input data
# Inputs:
#   eye_data_filename: File path where the eye measurements are saved in csv format (with headers: 
#                      "timestamp,left_palpebral_height,left_pupil_to_lower,right_palpebral_height,right_pupil_to_lower")
#   eye_images_folder: File path to folder where eye images are saved 
# Outputs:
#   None
    if testGUI:
        import eyespy_gui as gui
        import tkinter as tk
        import matplotlib
        matplotlib.use("TkAgg")  # Ensure Matplotlib integrates with Tkinter

        print("GUI is currently in progress - Numbers may not be correct")
        root = tk.Tk(baseName="GUI")
        print("Created Root")
        app = gui.EyeTrackingGUI(root, eye_images_folder, eye_data_filename) # Will need to add Data filename so its not random data
        root.mainloop()
    
    else:
        print(f"display_GUI_from_data function is currently blank")

        print(f"The Eye Data is Saved in: {eye_data_filename}")
        print(f"The Eye Images are in: {eye_images_folder}")

    return

""" Main Section of Code """
# testing
if testOnlyGUI:
    display_GUI_from_data(None, "./eye_tracking_output/")
    exit


# for now get data from a file
# Recommended: file_name = './Rachel_120fps_1080p.mov'

# collect video and output predicted measurements and frames for each eye
csv_file, folder_with_images = GUI_to_get_data(testing=False,saving=True)
#updated_csv_file = None
#folder_with_images = "./eye_tracking_output/"

# testing if this is working
# plt.imshow(base_images[1,:])

# Change the resolution of the image to match the resolution that the ML model expects
folder_with_cropped_frames = change_resolution_frames(folder_with_images, saving=True, testing=False)

# Put the images through the NN and get the output parameter predictions
updated_csv_file = apply_ML_model(csv_file, folder_with_cropped_frames)

# Plug data into GUI to display to the doctor
"""Source of Error - can't open GUI window when live video window from the GUI_to_get_data is also open: 
https://stackoverflow.com/questions/24274072/tkinter-pyimage-doesnt-exist """
display_GUI_from_data(updated_csv_file, folder_with_images)

