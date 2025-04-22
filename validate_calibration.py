import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Load the annotated ground truth data
annotated_df = pd.read_csv('eye_frames_20250414_124415/avneesh_annotated.csv')

# Load the measurements from our script
measurements_df = pd.read_csv('eye_frames_20250414_133947/eye_measurements.csv')

print(f'Annotated data: {len(annotated_df)} samples')
print(f'Measured data: {len(measurements_df)} samples')

# Extract frame numbers from filenames
def extract_frame_number(filename):
    match = re.search(r'avneesh_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

annotated_df['frame'] = annotated_df['Filename'].apply(extract_frame_number)

# Add eye side information
annotated_df['eye_side'] = annotated_df['Filename'].apply(lambda x: 'left' if 'left_eye' in x else 'right')

# Prepare for comparison
left_annotated = annotated_df[annotated_df['eye_side'] == 'left'].sort_values('frame')
right_annotated = annotated_df[annotated_df['eye_side'] == 'right'].sort_values('frame')

# Check frame ranges
print(f'Annotated frames range: {annotated_df["frame"].min()} to {annotated_df["frame"].max()}')
print(f'Measured frames range: {measurements_df["frame"].min()} to {measurements_df["frame"].max()}')

# Create a merged dataset for comparison
comparison_data = []

for _, row in annotated_df.iterrows():
    frame = row['frame']
    if frame is not None and frame in measurements_df['frame'].values:
        measured = measurements_df[measurements_df['frame'] == frame].iloc[0]
        
        if row['eye_side'] == 'left':
            comparison_data.append({
                'frame': frame,
                'eye_side': 'left',
                'vph_annotated': row['vpf_expected'],
                'vph_measured_px': measured['left_palpebral_height_px'],
                'vph_measured': measured['left_palpebral_height_mm'],
                'mrd1_annotated': row['mrd1_expected'],
                'mrd1_measured_px': measured['left_pupil_to_lower_px'],
                'mrd1_measured': measured['left_pupil_to_lower_mm']
            })
        else:
            comparison_data.append({
                'frame': frame,
                'eye_side': 'right',
                'vph_annotated': row['vpf_expected'],
                'vph_measured_px': measured['right_palpebral_height_px'],
                'vph_measured': measured['right_palpebral_height_mm'],
                'mrd1_annotated': row['mrd1_expected'],
                'mrd1_measured_px': measured['right_pupil_to_lower_px'],
                'mrd1_measured': measured['right_pupil_to_lower_mm']
            })

comparison_df = pd.DataFrame(comparison_data)

# Calculate error metrics
if len(comparison_df) > 0:
    current_calibration = measurements_df["calibration_factor"].mean()
    print(f'\nMatched {len(comparison_df)} frames for comparison')
    print(f'Current calibration factor: {current_calibration:.5f} mm/px')
    
    # Calculate average ratio between annotated and measured values
    vph_ratio = comparison_df['vph_annotated'].mean() / comparison_df['vph_measured'].mean()
    mrd1_ratio = comparison_df['mrd1_annotated'].mean() / comparison_df['mrd1_measured'].mean()
    
    print(f'VPH ratio (annotated/measured): {vph_ratio:.2f}')
    print(f'MRD1 ratio (annotated/measured): {mrd1_ratio:.2f}')
    
    # Calculate a better calibration factor
    suggested_vph_calibration = current_calibration * vph_ratio
    suggested_mrd1_calibration = current_calibration * mrd1_ratio
    suggested_calibration = (suggested_vph_calibration + suggested_mrd1_calibration) / 2
    
    print(f'Suggested calibration factor: {suggested_calibration:.5f} mm/px')
    print(f'Pupil diameter for this calibration: {suggested_calibration / current_calibration * 11.7:.2f} mm')
    
    # Recalculate measurements with suggested calibration
    comparison_df['vph_adjusted'] = comparison_df['vph_measured_px'] * suggested_calibration
    comparison_df['mrd1_adjusted'] = comparison_df['mrd1_measured_px'] * suggested_calibration

    # Calculate error metrics for original calibration
    comparison_df['vph_error'] = comparison_df['vph_measured'] - comparison_df['vph_annotated']
    comparison_df['mrd1_error'] = comparison_df['mrd1_measured'] - comparison_df['mrd1_annotated']
    
    # Calculate error metrics for adjusted calibration
    comparison_df['vph_adjusted_error'] = comparison_df['vph_adjusted'] - comparison_df['vph_annotated']
    comparison_df['mrd1_adjusted_error'] = comparison_df['mrd1_adjusted'] - comparison_df['mrd1_annotated']
    
    # Error statistics for original calibration
    print('\nOriginal Calibration Error:')
    print('VPH Error (mm):')
    print(f'  Mean absolute error: {comparison_df["vph_error"].abs().mean():.2f}')
    print(f'  Root mean squared error: {np.sqrt((comparison_df["vph_error"] ** 2).mean()):.2f}')
    
    print('MRD1 Error (mm):')
    print(f'  Mean absolute error: {comparison_df["mrd1_error"].abs().mean():.2f}')
    print(f'  Root mean squared error: {np.sqrt((comparison_df["mrd1_error"] ** 2).mean()):.2f}')
    
    # Error statistics for adjusted calibration
    print('\nAdjusted Calibration Error:')
    print('VPH Error (mm):')
    print(f'  Mean absolute error: {comparison_df["vph_adjusted_error"].abs().mean():.2f}')
    print(f'  Root mean squared error: {np.sqrt((comparison_df["vph_adjusted_error"] ** 2).mean()):.2f}')
    
    print('MRD1 Error (mm):')
    print(f'  Mean absolute error: {comparison_df["mrd1_adjusted_error"].abs().mean():.2f}')
    print(f'  Root mean squared error: {np.sqrt((comparison_df["mrd1_adjusted_error"] ** 2).mean()):.2f}')
    
    # Display sample comparison with both calibrations
    print('\nSample comparison:')
    sample_df = comparison_df[['frame', 'eye_side', 'vph_annotated', 'vph_measured', 'vph_adjusted', 
                             'mrd1_annotated', 'mrd1_measured', 'mrd1_adjusted']].head(10)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(sample_df.to_string(index=False, float_format='%.2f'))

    # Create scatter plot for VPH with both calibrations
    plt.figure(figsize=(12, 10))
    
    # Original VPH calibration
    plt.subplot(2, 2, 1)
    plt.scatter(comparison_df['vph_annotated'], comparison_df['vph_measured'], alpha=0.6, label='Original')
    plt.plot([0, 10], [0, 10], 'r--')  # Perfect correlation line
    plt.xlabel('Annotated VPH (mm)')
    plt.ylabel('Measured VPH (mm)')
    plt.title('Original Calibration - VPH')
    plt.grid(True)
    plt.legend()
    
    # Adjusted VPH calibration
    plt.subplot(2, 2, 2)
    plt.scatter(comparison_df['vph_annotated'], comparison_df['vph_adjusted'], alpha=0.6, color='green', label='Adjusted')
    plt.plot([0, 10], [0, 10], 'r--')  # Perfect correlation line
    plt.xlabel('Annotated VPH (mm)')
    plt.ylabel('Adjusted VPH (mm)')
    plt.title('Adjusted Calibration - VPH')
    plt.grid(True)
    plt.legend()
    
    # Original MRD1 calibration
    plt.subplot(2, 2, 3)
    plt.scatter(comparison_df['mrd1_annotated'], comparison_df['mrd1_measured'], alpha=0.6, label='Original')
    plt.plot([0, 5], [0, 5], 'r--')  # Perfect correlation line
    plt.xlabel('Annotated MRD1 (mm)')
    plt.ylabel('Measured MRD1 (mm)')
    plt.title('Original Calibration - MRD1')
    plt.grid(True)
    plt.legend()
    
    # Adjusted MRD1 calibration
    plt.subplot(2, 2, 4)
    plt.scatter(comparison_df['mrd1_annotated'], comparison_df['mrd1_adjusted'], alpha=0.6, color='green', label='Adjusted')
    plt.plot([0, 5], [0, 5], 'r--')  # Perfect correlation line
    plt.xlabel('Annotated MRD1 (mm)')
    plt.ylabel('Adjusted MRD1 (mm)')
    plt.title('Adjusted Calibration - MRD1')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('calibration_comparison.png')
    print('\nCreated comparison plot: calibration_comparison.png')

else:
    print('No matching frames found for comparison') 