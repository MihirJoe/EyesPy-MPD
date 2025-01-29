
""" Import Libraries Here """




""" Write Functions Here """ 

def get_data_from_file(file_name):
# Function that takes in a fileName and outputs an image or array of images
# Input: 
#   file_name - A string with the complete file path
# Output: 
#   images - Array of images (all frames if video input) from the file

    images = 0

    return images

def isolate_eye_images(images):
# Function that takes in an array of face images and outputs two arrays of eye images (one for each eye)
# Input: 
#   images - An array of images (one for each video frame)
# Outputs: 
#   right_eye_images: array of isolated, cropped image of the right eye for every frame
#   left_eye_images: array of isolated, cropped image of the left eye for every frame
    right_eye_images = 0
    left_eye_images = 0

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
file_name = './download_left_eye.jpg'

# get frames as an array of images
base_images = get_data_from_file(file_name)

# get isolated images for each eye for each frame
right_eye_frames, left_eye_frames = isolate_eye_images(base_images)

# Change the resolution of the image to match the resolution that the ML model expects
right_eye_frames = change_resolution(right_eye_frames)
left_eye_frames = change_resolution(left_eye_frames)

# Put the images through the NN and get the output parameter predictions
right_eye_measurements = apply_ML_model(right_eye_frames)
left_eye_measurements = apply_ML_model(left_eye_frames)

# Plug data into GUI to display to the doctor
display_GUI_from_data(right_eye_measurements, left_eye_measurements, right_eye_frames, left_eye_frames)

