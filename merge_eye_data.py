import pandas as pd
import re
import os
import numpy as np

# Read the actual measurements data
actual_df = pd.read_csv("eye_frames_20250414_124415/actual_eye_measurements.csv")

# Read the annotated data
annotated_df = pd.read_csv("eye_frames_20250414_124415/avneesh_annotated.csv")

# Extract frame numbers from filenames in annotated data
def extract_frame(filename):
    match = re.search(r'avneesh_(\d+)_(left|right)_eye\.jpg', filename)
    if match:
        return int(match.group(1)), match.group(2)
    return None, None

# Create new columns for frame number and eye side
annotated_df['frame_num'] = annotated_df['Filename'].apply(lambda x: extract_frame(x)[0])
annotated_df['eye_side'] = annotated_df['Filename'].apply(lambda x: extract_frame(x)[1])

# Create separate dataframes for left and right eyes
left_eye_df = annotated_df[annotated_df['eye_side'] == 'left'].rename(
    columns={'vpf_expected': 'left_vpf_expected', 'mrd1_expected': 'left_mrd1_expected'})
right_eye_df = annotated_df[annotated_df['eye_side'] == 'right'].rename(
    columns={'vpf_expected': 'right_vpf_expected', 'mrd1_expected': 'right_mrd1_expected'})

# Drop unnecessary columns
left_eye_df = left_eye_df[['frame_num', 'left_vpf_expected', 'left_mrd1_expected']]
right_eye_df = right_eye_df[['frame_num', 'right_vpf_expected', 'right_mrd1_expected']]

# Merge left and right eye data based on frame number
merged_annotated = pd.merge(left_eye_df, right_eye_df, on='frame_num', how='outer')

# Convert frame numbers to integer
actual_df['frame'] = actual_df['frame'].apply(lambda x: int(x) if pd.notnull(x) else np.nan)
merged_annotated['frame_num'] = merged_annotated['frame_num'].astype('Int64')  # nullable integer type

# Now merge with actual measurements
# Select only the relevant columns from actual_df
actual_selected = actual_df[['frame', 'left_palpebral_height', 'right_palpebral_height', 
                          'left_pupil_to_upper', 'right_pupil_to_upper']]

# Print some debug information
print(f"Actual measurements frame types: {actual_selected['frame'].dtype}")
print(f"Annotated frame types: {merged_annotated['frame_num'].dtype}")
print(f"Actual measurements frame sample: {actual_selected['frame'].head()}")
print(f"Annotated frame sample: {merged_annotated['frame_num'].head()}")

# Merge with annotated data
final_df = pd.merge(actual_selected, merged_annotated, left_on='frame', right_on='frame_num', how='outer')

# Clean up the final dataframe
if 'frame_num' in final_df.columns:
    final_df = final_df.drop('frame_num', axis=1)

# Sort by frame number
final_df = final_df.sort_values('frame')

# Save the merged data
output_path = "eye_frames_20250414_124415/merged_eye_measurements.csv"
final_df.to_csv(output_path, index=False)

print(f"Merged data saved to {output_path}")
print(f"Total rows: {len(final_df)}")
print(f"Number of actual measurement rows: {len(actual_df)}")
print(f"Number of annotated rows (pairs): {len(merged_annotated)}")
print(f"Rows with both actual and annotated data: {len(final_df.dropna(subset=['left_palpebral_height', 'left_vpf_expected']))}") 