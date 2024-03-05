from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm
import cv2
import os
import re

def visualize_image_scanpath(input_image_path, generated_scanpaths, prefix="scanpath_prediction_", path_to_save="./outputs", overlay_image_path=None, overlay_alpha=0.4, _base_name=''):
    """
    Visualizes a scanpath on an image, with an option to overlay a second image transparently.

    Parameters:
    - input_image_path: str, path to the input image.
    - generated_scanpaths: list of lists, each sublist contains scanpath coordinates as [x1, y1, x2, y2, ...].
    - prefix: str, prefix for the saved file name.
    - path_to_save: str, directory where the visualization will be saved.
    - overlay_image_path: str, optional, path to a second image to overlay transparently.
    - overlay_alpha: float, optional, transparency level for the overlay image.
    """
    # Load the original image
    if input_image_path:
        original_image = mpimg.imread(input_image_path)
    else:
        original_image = np.ones((720, 1280, 3), dtype=np.uint8) * 255

    # Ensure the output directory exists
    os.makedirs(path_to_save, exist_ok=True)

    # Get the file name without extension from the input image path
    if input_image_path:
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    else:
        base_name=_base_name

    # Find the next available file number
    existing_files = os.listdir(path_to_save)
    pattern = re.compile(rf'^{prefix}{base_name}_(\d+)\.png$')
    existing_numbers = [int(pattern.match(f).group(1)) for f in existing_files if pattern.match(f)]
    next_number = 1
    if existing_numbers:
        next_number = max(existing_numbers) + 1

    # Setup and create a single subplot
    DPI = 80
    marker_size = original_image.shape[1] / 50  # Adjusted marker size
    line_width = original_image.shape[1] / 500  # Adjusted line width
    fig, axs = plt.subplots(1, 1, figsize=(original_image.shape[1] / DPI, original_image.shape[0] / DPI))

    # Adjust subplot to remove borders and fill the figure
    axs.set_position([0, 0, 1, 1])

    # Display the original image in the background
    axs.imshow(original_image)

    # If an overlay image path is provided, load, resize, and overlay it
    if overlay_image_path:
        overlay_image = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)
        # Resize overlay to match the input image size
        overlay_resized = cv2.resize(overlay_image, (original_image.shape[1], original_image.shape[0]))
        # Convert resized overlay image to RGBA for proper matplotlib handling
        overlay_resized_rgba = cv2.cvtColor(overlay_resized, cv2.COLOR_BGR2RGBA)
        axs.imshow(overlay_resized_rgba, alpha=overlay_alpha)

    axs.axis('off')

    # Separate the coordinates for scanpath visualization
    points_x = [generated_scanpaths[0][i+1] * original_image.shape[1] for i in range(0, len(generated_scanpaths[0]), 2)]
    points_y = [generated_scanpaths[0][i] * original_image.shape[0] for i in range(0, len(generated_scanpaths[0]), 2)]
    colors = cm.rainbow(np.linspace(0, 1, len(points_x)))

    # Draw points and lines between them
    previous_point = None
    for x, y, c in zip(points_x, points_y, colors):
        if previous_point is not None:
            axs.plot([x, previous_point[0]], [y, previous_point[1]], color='blue', linewidth=line_width, alpha=0.35)
        previous_point = [x, y]
        axs.plot(x, y, marker='o', markersize=marker_size, color=c, alpha=.8)

    # Save the figure
    plt.savefig(f"{path_to_save}/{prefix}{base_name}_{next_number}.png")
    plt.clf()
    plt.close('all')
    # print("Scanpath image printed with optional overlay.")

def visualize_video_scanpath(video_frames_folder, scanpaths, path_to_save, prefix="scanpath_video_frame_", overlay_folder=None, overlay_alpha=0.4, history_length=10, generated=True):
    """
    Visualizes scanpath on a sequence of video frames, showing the current point and the history of the last N points based on provided scanpaths structure.

    Parameters:
    - video_frames_folder: str, optional, path to the folder containing video frames.
    - scanpaths: list of lists, where the inner list contains all x, y coordinates for the entire video sequence.
    - prefix: str, prefix for the saved file names.
    - path_to_save: str, directory where the visualized frames will be saved.
    - overlay_folder: str, optional, path to the folder containing overlay images for each frame.
    - overlay_alpha: float, optional, transparency level for the overlay image.
    - history_length: int, the number of previous points to include in the visualization along with the current point.
    """
    # Ensure the output directory exists
    os.makedirs(path_to_save, exist_ok=True)

    # Flatten the list of scanpaths if it's nested
    flattened_scanpaths = [val for sublist in scanpaths for val in sublist]

    if generated:
        # Get a sorted list of video frame files
        video_frames = sorted([f for f in os.listdir(video_frames_folder) if f.endswith('.png') or f.endswith('.jpg')])

        # Iterate through each frame with tqdm for a progress bar
        for idx, frame_file in tqdm(enumerate(video_frames), total=len(video_frames), desc="Processing frames", unit="frame", ncols=100):
            input_image_path = os.path.join(video_frames_folder, frame_file)

            # Calculate the start index for the scanpath history
            start_idx = max(0, idx * 2 - history_length * 2)
            end_idx = idx * 2 + 2  # Ensure we include the current point
            current_scanpath = flattened_scanpaths[start_idx:end_idx]

            # If there's an overlay folder provided, prepare the overlay image path for the current frame
            if overlay_folder:
                overlay_image_path = os.path.join(overlay_folder, frame_file)  # Assuming same naming convention
                if not os.path.exists(overlay_image_path):
                    overlay_image_path = None  # Handle missing overlay image gracefully
            else:
                overlay_image_path = None

            # Generate a filename for the output image
            base_name = os.path.splitext(frame_file)[0]
            output_file_name = f"{prefix}{base_name}.png"

            # Visualize the current and previous scanpath points on the image
            visualize_image_scanpath(input_image_path, [current_scanpath], prefix, path_to_save, overlay_image_path, overlay_alpha)
    else:
        input_image_path = None
        # Get a sorted list of video frame files
        if video_frames_folder:
            video_frames = sorted([f for f in os.listdir(video_frames_folder) if f.endswith('.png') or f.endswith('.jpg')])
        if overlay_folder:
            video_frames_overlay = sorted([f for f in os.listdir(overlay_folder) if f.endswith('.png') or f.endswith('.jpg')])
        
        # Iterate through each frame with tqdm for a progress bar
        for idx, frame_file in tqdm(enumerate(scanpaths), total=len(scanpaths), desc="Processing frames", unit="frame", ncols=100):
            if video_frames_folder:
                input_image_path = os.path.join(video_frames_folder, video_frames[idx])
            # Calculate the start index for the scanpath history
            start_idx = max(0, idx * 2 - history_length * 2)
            end_idx = idx * 2 + 2  # Ensure we include the current point
            current_scanpath = flattened_scanpaths[start_idx:end_idx]

            # If there's an overlay folder provided, prepare the overlay image path for the current frame
            if overlay_folder:
                overlay_image_path = os.path.join(overlay_folder, video_frames_overlay[idx])  # Assuming same naming convention
                if not os.path.exists(overlay_image_path):
                    overlay_image_path = None  # Handle missing overlay image gracefully
            else:
                overlay_image_path = None

            # Generate a filename for the output image
            if video_frames_folder:
                base_name = os.path.splitext(video_frames[idx])[0]
            else:
                base_name = str(idx).zfill(5)
            output_file_name = f"{prefix}{base_name}.png"

            # Visualize the current and previous scanpath points on the image
            visualize_image_scanpath(input_image_path, [current_scanpath], prefix, path_to_save, overlay_image_path, overlay_alpha, _base_name = base_name)