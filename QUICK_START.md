# Quick Start Guide for EyesPy Iris Segmentation

This guide will help you get started with the EyesPy iris segmentation tool.

## Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv eyespy
   source eyespy/bin/activate  # On Windows: eyespy\Scripts\activate
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models**:
   - Get the SAM model: [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
   - Get the dlib face predictor: [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) (extract after downloading)
   
   Place these files in the project directory or in a `weights/` subdirectory.

## Running Iris Segmentation

### Command Line

The simplest way to use EyesPy is through the command line:

```bash
python iris_segmentation.py
```

When prompted, enter the path to your image. Results will be saved to the `mihir_test/` directory.

### Python API

```python
from iris_segmentation import IrisSegmentor

# Initialize the segmentor
segmentor = IrisSegmentor()

# Process a single image
results = segmentor.process_image("path/to/your/image.jpg")

# Access the results
if results:
    for eye_key, eye_data in results.items():
        if 'diameter' in eye_data:
            print(f"Eye: {eye_key}")
            print(f"Iris diameter: {eye_data['diameter']:.2f} pixels")
            print(f"Estimated diameter: {eye_data['diameter_mm']:.2f} mm")
            print(f"Mask path: {eye_data['mask_path']}")
            print(f"Visualization: {eye_data['visualization_path']}")
            print("---")
```

## Troubleshooting

### No eyes detected

If the system fails to detect eyes:

1. Make sure the face is clearly visible (for full-face images)
2. For close-up images, ensure good lighting and contrast
3. Try using a different image if problems persist

### Poor segmentation results

If iris segmentation quality is not satisfactory:

1. Ensure the image has good lighting and clear iris boundaries
2. Check that the SAM model is loaded correctly
3. Experiment with different images or adjust parameters in the code

## Tips for Best Results

- **Image Quality**: Use high-resolution images with good lighting
- **Face Position**: For full-face images, ensure a clear frontal view
- **Eye Clarity**: Make sure the iris boundaries are visible
- **Manual Mode**: Use the interactive mode for challenging cases

## Output Files

The system generates several files for each processed image:

- `*_iris_mask.png`: Binary mask of the detected iris
- `*_visualization.png`: Original image with green overlay on the iris and blue circle showing the measured diameter
- Split images of left and right eyes (for close-up images with both eyes)

## Next Steps

After successfully running the basic segmentation, you can:

1. Experiment with different images
2. Modify the parameters in the code for specific use cases
3. Integrate the system into your own applications 