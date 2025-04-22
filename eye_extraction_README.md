# Eye Frame Extraction and Measurement

This README provides instructions for using the eye extraction and measurement scripts.

## Overview

The provided scripts allow you to:

1. Extract left and right eye frames from a video
2. Save the eye frames to designated folders
3. Measure eye parameters (palpebral height and pupil-to-lid distances)
4. Save the measurements to a CSV file
5. Visualize the results with the included video player

## Files

- `eye_frame_extractor.py`: Core functionality for eye extraction and measurement
- `eye_frame_extractor_calibrated.py`: Enhanced version with SAM model integration for improved cornea detection
- `extract_and_measure_eyes.py`: Simple script that demonstrates full pipeline usage
- `extract_and_measure_eyes_calibrated.py`: Enhanced script with SAM model support

## Dependencies

The scripts require:
- OpenCV (cv2)
- NumPy
- Pandas
- dlib (with the shape_predictor_68_face_landmarks.dat file)
- Tkinter (for visualization)

### SAM Model Dependencies (for enhanced cornea detection)
- PyTorch
- segment-anything (SAM)
- torchvision
- PIL

## Installation of SAM Model

To use the SAM model for enhanced cornea detection:

1. Install the required dependencies:
```bash
pip install torch torchvision segment-anything
```

2. Download the SAM model checkpoint from Meta AI:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Usage

### Basic Eye Extraction and Measurement

For the simplest use case:

```bash
python extract_and_measure_eyes.py /path/to/your/video.mp4
```

This will:
1. Process the video
2. Extract eye frames
3. Measure eye parameters
4. Display the results in a video player

### Advanced Usage with SAM Model

You can use the enhanced extraction script with the SAM model for improved cornea detection:

```bash
# Extract eye frames and measurements with SAM model
python eye_frame_extractor_calibrated.py /path/to/your/video.mp4 --sam-checkpoint /path/to/sam_vit_h_4b8939.pth

# Extract with SAM model and specify custom cornea diameter (default is 11.8mm)
python eye_frame_extractor_calibrated.py /path/to/your/video.mp4 --sam-checkpoint /path/to/sam_vit_h_4b8939.pth --cornea-diameter 11.8

# Extract but don't save the frames (measurements only)
python eye_frame_extractor_calibrated.py /path/to/your/video.mp4 --no-save --sam-checkpoint /path/to/sam_vit_h_4b8939.pth

# Process previously extracted frames
python eye_frame_extractor_calibrated.py /path/to/your/video.mp4 --process-only --output previously_extracted_eyes

# Visualize SAM cornea detection for a specific frame
python eye_frame_extractor_calibrated.py /path/to/your/video.mp4 --sam-checkpoint /path/to/sam_vit_h_4b8939.pth --visualize-frame 3
```

### Using the GUI with SAM Model

```bash
python extract_and_measure_eyes_calibrated.py /path/to/your/video.mp4 --sam-checkpoint /path/to/sam_vit_h_4b8939.pth
```

## Output

The scripts generate:

1. A folder with extracted eye frames:
   - `left_eye/`: Contains frames of the left eye
   - `right_eye/`: Contains frames of the right eye

2. A CSV file with eye measurements:
   - Contains columns for timestamp, frame number, and eye measurements
   - Measurements include:
     - `left_palpebral_height`: Vertical height of left eye opening
     - `left_pupil_to_lower`: Distance from pupil to lower lid (left eye)
     - `right_palpebral_height`: Vertical height of right eye opening
     - `right_pupil_to_lower`: Distance from pupil to lower lid (right eye)
     - `cornea_diameter`: Cornea diameter (calibrated to 11.8mm)

## Integration with Existing Codebase

To use these measurements with the existing software pipeline:

```bash
python software_pipeline.py --eye-data /path/to/eye_measurements.csv --video /path/to/your/video.mp4
```

## SAM Model Advantages

Using the SAM (Segment Anything Model) for cornea detection offers several advantages:

1. **Higher accuracy**: SAM provides more precise segmentation of the cornea compared to traditional methods
2. **Robustness**: Better performance in challenging lighting conditions and for diverse eye shapes
3. **Consistent calibration**: More reliable cornea diameter measurements for accurate calibration
4. **Integration with existing pipeline**: Seamlessly works with the current workflow by enhancing cornea detection

## SAM Visualization Output

When using the `--visualize-frame` option, the script will generate several output files for the specified frame:

1. **Original images**: 
   - `{left/right}_frame_{number}_original.jpg` - The original eye region image

2. **SAM mask visualization**:
   - `{left/right}_frame_{number}_cornea.jpg` - Eye image with cornea mask overlay, contour, and center point
   - `{left/right}_frame_{number}_cornea_mask.jpg` - The binary mask produced by SAM for the cornea
   - `{left/right}_frame_{number}_edges.jpg` - The edge detection used to help identify the cornea

The visualization files are saved in the `sam_visualizations` subdirectory within the output directory.

### Visualization color coding:
- **Red overlay**: The cornea mask produced by SAM
- **Green contour**: The detected contour of the cornea
- **Blue circle**: The enclosing circle used to determine cornea diameter
- **Yellow dot**: The center point of the cornea

These visualizations help you understand how the SAM model is detecting and segmenting the cornea in each eye frame. 