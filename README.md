# EyesPy-MPD: Eye Measurement and Analysis Tool

EyesPy-MPD is a Python-based application designed for ophthalmologists and medical professionals to analyze and measure eye parameters from video input. The tool uses computer vision and machine learning techniques to accurately detect facial landmarks, measure eye features, and visualize results.

## Features

- **Video Processing**: 
  - Upload pre-recorded videos or capture live video for analysis
  - Real-time eye detection and measurement overlay
  - Frame-by-frame navigation for detailed review

- **Eye Measurements**:
  - Palpebral height (eyelid opening distance)
  - Pupil to lower eyelid distance
  - Automatic facial landmark detection using dlib

- **Data Visualization**:
  - Real-time graphing of measurements
  - Interactive plots with clear legends
  - Comparative analysis of left and right eye metrics

- **Export Capabilities**:
  - Save measurements to CSV for further analysis
  - Export processed frames with measurement overlays

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Webcam (for live video capture)
- shape_predictor_68_face_landmarks.dat file (for dlib facial landmark detection)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/EyesPy-MPD.git
   cd EyesPy-MPD
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the shape predictor file:
   The application requires the `shape_predictor_68_face_landmarks.dat` file for facial landmark detection. This file must be placed in the project's root directory.
   
   You can download it from:
   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   
   After downloading, extract the file and place it in the project root directory.

### Usage

#### Running the Simplified GUI (Recommended)

```
python eyeMeasure2.py
```

This launches a streamlined interface with:
- Video upload button
- Playback controls
- Interactive measurement display
- Real-time graphing

#### Running the Full Pipeline

```
python software_pipeline.py
```

The full pipeline includes:
- Video capture/upload
- Eye isolation
- Resolution standardization
- Optional ML model processing
- Comprehensive visualization

## Application Interface

### Video Controls
- **Upload Video**: Load a pre-recorded video file
- **Play/Pause**: Control video playback
- **Frame Slider**: Navigate through video frames
- **Export CSV**: Save measurements to file

### Measurement Display
- Real-time measurements for both eyes
- Graph showing measurement trends over time
- Legend positioned for optimal visibility

## Development

### Project Structure
- `eyeMeasure2.py`: Streamlined GUI application
- `software_pipeline.py`: Full processing pipeline
- `left_and_right_eye_measurements.py`: Core eye tracking functionality
- `eyespy_mpd_NN.py`: Neural network model for enhanced measurements

### Adding New Features
To extend the application, you can:
1. Modify the `EyeTracker` class to add new measurement types
2. Enhance the GUI by adding new visualization options
3. Implement additional data export formats

## Troubleshooting

### Common Issues
- **Video Upload Errors**: Ensure video codecs are compatible with OpenCV
- **Landmark Detection Failures**: Check lighting conditions and face visibility
- **Performance Issues**: For high-resolution videos, consider using a more powerful machine

### Getting Help
If you encounter any issues, please:
1. Check the troubleshooting section
2. Look for similar issues in the project issues
3. Create a new issue with detailed information about your problem

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- dlib for the facial landmark detection
- OpenCV for image processing capabilities
- Tkinter for the GUI framework 