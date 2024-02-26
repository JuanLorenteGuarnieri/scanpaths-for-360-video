import numpy as np
import cv2
import os

def generate_image_random_scanpath(length=10):
    """
    Generates a randomized scanpath.

    Parameters:
    - length: int, the number of points in the scanpath.

    Returns:
    - A list of normalized coordinates representing the scanpath.
    """
    # Generate random points between 0 and 1
    # Each point consists of an x and y coordinate, hence length*2
    scanpath = np.random.rand(length*2).tolist()

    return scanpath

def generate_video_random_scanpath(video_frames_folder):
    """
    Generates a random scanpath for a sequence of video frames.

    Parameters:
    - video_frames_folder: str, path to the folder containing video frames.

    Returns:
    - A list of scanpaths, each representing the sequence of normalized coordinates for each frame in the video.
    """
    # Get a sorted list of video frame files
    video_frames = sorted([f for f in os.listdir(video_frames_folder) if f.endswith('.png') or f.endswith('.jpg')])

    # Initialize a list to hold the scanpath for each frame
    result = []
    video_scanpath = []

    for _ in video_frames:
        # Generate a random scanpath for the current frame
        frame_scanpath = generate_image_random_scanpath(1)
        video_scanpath.extend(frame_scanpath)

    result.append(video_scanpath)

    return result


def generate_image_saliency_scanpath(saliency_map_path, length=10):
    """
    Generates a scanpath based on a saliency map, starting with the maximum saliency point.

    Parameters:
    - saliency_map_path: str, path to the saliency map image (grayscale).
    - length: int, the number of points in the scanpath.

    Returns:
    - A list of coordinates representing the scanpath, normalized to [0, 1].
    """
    # Load the saliency map as a grayscale image
    saliency_map = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)
    if saliency_map is None:
        raise ValueError("Saliency map could not be loaded.")

    scanpath = []
    for _ in range(length):
        # Find the value and location of the maximum saliency
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(saliency_map)

        # Normalize the coordinates of the maximum saliency point
        y_normalized, x_normalized = max_loc[0] / saliency_map.shape[1], max_loc[1] / saliency_map.shape[0]
        scanpath.extend([x_normalized, y_normalized])


    return scanpath

def generate_video_saliency_scanpath(saliency_map_folder):
    """
    Generates a scanpath based on a sequence of saliency maps, such as from a video, starting with the maximum saliency point for each frame.

    Parameters:
    - saliency_map_folder: str, path to the folder containing saliency map images (grayscale).

    Returns:
    - A list of scanpaths, each representing the sequence of normalized coordinates for each frame in the video.
    """
    # Get a sorted list of saliency map files
    saliency_maps = sorted([f for f in os.listdir(saliency_map_folder) if f.endswith('.png') or f.endswith('.jpg')])

    # Initialize a list to hold the scanpath for each frame
    result = []
    video_scanpath = []

    for map_file in saliency_maps:
        map_path = os.path.join(saliency_map_folder, map_file)
        saliency_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if saliency_map is None:
            raise ValueError(f"Saliency map {map_file} could not be loaded.")

        frame_scanpath = generate_image_saliency_scanpath(map_path, length=1)
        video_scanpath.extend(frame_scanpath)
    result.append(video_scanpath)

    return result