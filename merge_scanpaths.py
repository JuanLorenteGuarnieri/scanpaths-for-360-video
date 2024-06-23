import os
import csv
import json

def process_file(file_path):
    """
    Processes a single CSV file to extract and group scanpaths based on frame differences.

    This function reads a CSV file containing gaze data, where each row represents a gaze point
    with associated frame information. The function groups gaze points into scanpaths
    based on the difference between consecutive frames, and applies specific rules to adjust 
    the grouping to ensure consistency.

    Parameters:
    file_path (str): The path to the CSV file to be processed.

    Returns:
    list: A list of scanpath groups, where each group contains up to 10 scanpaths. 
          Each scanpath is represented as a list of gaze points, with each point being a 
          list of two float values corresponding to the 'v' and 'u' coordinates.
    """
    grouped_data = []
    current_group = []
    previous_frame = -1
    last_saved_frame = -8
    adjustment_made = False

    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            current_frame = int(row['frame'])

            # Check if the current frame is different from the previous frame
            if current_frame != previous_frame:
                if previous_frame > 0 and current_frame < previous_frame:
                    grouped_data.append(current_group)
                    current_group = []
                    previous_frame = -1
                    last_saved_frame = -8
                    adjustment_made = False

                difference = current_frame - last_saved_frame

                # Check if the difference is greater than or equal to 8 or if an adjustment is needed
                if difference >= 8 or (difference > 0 and not adjustment_made):
                    current_group.append([float(row['v']), float(row['u'])])
                    if not adjustment_made and difference > 0 and difference < 8:
                        last_saved_frame = current_frame
                        adjustment_made = True
                    elif adjustment_made:
                        last_saved_frame += 8
                    else:
                        last_saved_frame = current_frame

            previous_frame = current_frame

        # Append the last group if it's not empty
        if current_group:
            grouped_data.append(current_group)

    return grouped_data[:10]  # Return only the first 10 groups

# Directory of the files
directory = ''  # CHANGE TO THE DIRECTORY WHERE THE .csv FILES OF THE D-SAV360 DATASET ARE STORED

# Specific file numbers to process
specific_numbers = ['0002', '0011', '1005', '1016', '2006', '2017', '1004', '5010', '5007', '5035']

# Collect 10 scanpaths from each file
total_scanpaths = []

# Process only the specified files
for number in specific_numbers:
    file_name = f'gaze_video_{number}.csv'
    full_path = os.path.join(directory, file_name)
    if os.path.exists(full_path):
        scanpaths = process_file(full_path)
        total_scanpaths.extend(scanpaths)
    else:
        print(f"The file {file_name} does not exist. Please check the path of the .csv files of the D-SAV360 dataset.")

# Save the total scanpaths to a .scanpaths file
save_path = 'output_scanpaths/ground_truth.scanpaths'
with open(save_path, 'w') as result_file:
    json.dump(total_scanpaths, result_file)

print("Process completed.")