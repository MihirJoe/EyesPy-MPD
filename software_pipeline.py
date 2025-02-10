
""" Import Libraries Here """
import os
import cv2
import numpy 

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
# Function that displays a GUI the patient interacts with to collect an image/video from a file OR via live feed from the camera
# Input: 
#   Saving - Saves each frame as a file if true
#   Testing - Prints helpful outputs
# Output: 
#   images - Array of images (all frames if video input) from the file
    # for now - request file name from screen
    file_name = './' + input(f"Enter the file name (must be in your current folder or a nested folder) of the video you would like to parse: ")

    images = get_data_from_file(file_name, testing=testing, saving=saving)
    return images

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

def change_resolution(images, ideal_shape = (256,256)):
# Function that takes an array of images as an input and updates the resolution of each image to make the ideal shape. 
# Uses blurring or reverse blurring/sharpening to decrease and increase resolution (respectively)
# Inputs:
#   images: An array of images
#   ideal_shape: The desired dimensions of each image. Has a default value to match the ML model expected input
# Outputs:  
#   resized_images: The image array where each image has been resized to the ideal_shape

    resized_images = images

    return resized_images

def apply_ML_model(images):
# Function that takes an array of images as an input, applies the ML model to each individual image/frame,
# and outputs the resulting data as an array
# Input: 
#   images: An array of images
# Output:
#   ml_predicted_values: An array of ML model predicted values corresponding to the array of images 
#                        (UPDATE TO INLCUDE PARAMETERS IN WHICH COLUMNS)

    ml_predicted_values = 0

    return ml_predicted_values

def display_GUI_from_data(right_eye_data, left_eye_data, right_eye_images, left_eye_images):
# Function to display the functional GUI with the input data
# Inputs:
#   right_eye_data: Array of predicted parameter values for the right eye images
#   left_eye_data: Array of predicted parameter values for the left eye images
#   right_eye_images: array of isolated, cropped images of the right eye for every frame
#   left_eye_images: array of isolated, cropped images of the left eye for every frame
# Outputs:
#   None


    return

""" Main Section of Code """


# for now get data from a file
# Recommended: file_name = './Rachel_120fps_1080p.mov'

# get frames as an array of images
base_images = GUI_to_get_data(testing=False,saving=True)

# testing if this is working
# plt.imshow(base_images[1,:])

# get isolated images for each eye for each frame
right_eye_frames, left_eye_frames = isolate_eye_images(base_images, saving=True, testing=False)

# Change the resolution of the image to match the resolution that the ML model expects
right_eye_frames = change_resolution(right_eye_frames)
left_eye_frames = change_resolution(left_eye_frames)

# Put the images through the NN and get the output parameter predictions
right_eye_measurements = apply_ML_model(right_eye_frames)
left_eye_measurements = apply_ML_model(left_eye_frames)

# Plug data into GUI to display to the doctor
display_GUI_from_data(right_eye_measurements, left_eye_measurements, right_eye_frames, left_eye_frames)

