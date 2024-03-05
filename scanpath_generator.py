import numpy as np
import cv2
import os
import config

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

def generate_image_random_saliency_scanpath(saliency_map_path, percentile, length=10):
    """
    Generates a scanpath based on a saliency map, selecting points randomly based on a given percentile.

    Parameters:
    - saliency_map_path: str, path to the saliency map image (grayscale).
    - percentile: int, the percentile to determine the saliency threshold (e.g., 50 for median).
    - length: int, the number of points in the scanpath.

    Returns:
    - A list of coordinates representing the scanpath, normalized to [0, 1].
    """
    # Load the saliency map as a grayscale image
    saliency_map = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)
    if saliency_map is None:
        raise ValueError("Saliency map could not be loaded.")

    threshold = 255 * percentile / 100

    # Identify all points above the threshold
    x_indices, y_indices = np.where(saliency_map > threshold)

    if len(x_indices) == 0:
        raise ValueError("No saliency points found above the threshold.")

    scanpath = []
    for _ in range(length):
        # Randomly select an index from those above the threshold
        idx = np.random.choice(range(len(x_indices)))
        x, y = x_indices[idx], y_indices[idx]

        # Normalize the coordinates
        y_normalized, x_normalized = y / saliency_map.shape[1], x / saliency_map.shape[0]
        scanpath.extend([x_normalized, y_normalized])

    return scanpath

def generate_video_random_saliency_scanpath(saliency_map_folder, percentile):
    """
    Generates a scanpath for each frame in a sequence of saliency maps, selecting points randomly based on a given percentile.

    Parameters:
    - saliency_map_folder: str, path to the folder containing saliency map images (grayscale).
    - percentile: int, the percentile to determine the saliency threshold (e.g., 50 for median).
    - length: int, the number of points in each frame's scanpath.

    Returns:
    - A list of scanpaths, each representing the sequence of normalized coordinates for each frame in the video.
    """
    # Get a sorted list of saliency map files
    saliency_maps = sorted([f for f in os.listdir(saliency_map_folder) if f.endswith('.png') or f.endswith('.jpg')])

    # Initialize a list to hold the scanpath for each frame
    video_scanpaths = []

    for map_file in saliency_maps:
        map_path = os.path.join(saliency_map_folder, map_file)
        # Use the generate_image_random_saliency_scanpath function for each saliency map
        frame_scanpath = generate_image_random_saliency_scanpath(map_path, percentile, 1)
        video_scanpaths.append(frame_scanpath)

    return video_scanpaths

def generate_image_probabilistic_saliency_scanpath(saliency_map_path, probabilistic_importance_factor=1.0):
    """
    Generates a scanpath based on a saliency map, selecting points probabilistically based on saliency values,
    adjusted by an importance factor.

    Parameters:
    - saliency_map_path: str, path to the saliency map image (grayscale).
    - probabilistic_importance_factor: float, factor to adjust the importance given to higher saliency values.

    Returns:
    - A list of coordinates representing the scanpath, normalized to [0, 1].
    """
    # Load the saliency map as a grayscale image
    saliency_map = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)
    if saliency_map is None:
        raise ValueError("Saliency map could not be loaded.")

    # Adjust the saliency values by the importance factor
    adjusted_saliency = np.power(saliency_map.flatten(), probabilistic_importance_factor)
    total_adjusted_saliency = np.sum(adjusted_saliency)
    if total_adjusted_saliency == 0:
        raise ValueError("Adjusted saliency results in zero total saliency.")
    probabilities = adjusted_saliency / total_adjusted_saliency

    # Choose indices based on the adjusted probability distribution
    chosen_indices = np.random.choice(len(adjusted_saliency), size=1, p=probabilities)

    # Convert flat indices back to 2D coordinates and normalize
    scanpath = []
    for idx in chosen_indices:
        x, y = divmod(idx, saliency_map.shape[1])  # Convert flat index to 2D coordinates
        y_normalized, x_normalized = y / saliency_map.shape[1], x / saliency_map.shape[0]
        scanpath.extend([x_normalized, y_normalized])

    return scanpath

def generate_video_probabilistic_saliency_scanpath(saliency_map_folder, probabilistic_importance_factor=1.0):
    """
    Generates a scanpath for each frame in a sequence of saliency maps, selecting points probabilistically based on saliency values.

    Parameters:
    - saliency_map_folder: str, path to the folder containing saliency map images (grayscale).

    Returns:
    - A list of scanpaths, each representing the sequence of normalized coordinates for each frame in the video.
    """
    # Get a sorted list of saliency map files
    saliency_maps = sorted([f for f in os.listdir(saliency_map_folder) if f.endswith('.png') or f.endswith('.jpg')])

    # Initialize a list to hold the scanpath for each frame
    video_scanpaths = []

    for map_file in saliency_maps:
        map_path = os.path.join(saliency_map_folder, map_file)
        # Use the generate_image_probabilistic_saliency_scanpath function for each saliency map
        frame_scanpath = generate_image_probabilistic_saliency_scanpath(map_path, probabilistic_importance_factor=probabilistic_importance_factor)
        video_scanpaths.append(frame_scanpath)

    return video_scanpaths

def apply_inhibition(saliency_map, points, radius, decay):
    """
    Apply inhibition of return to the saliency map around the given points with specified radius and decay.

    Parameters:
    - saliency_map: np.array, the saliency map.
    - points: list of tuples, points where inhibition will be applied.
    - radius: int, radius around points to apply inhibition.
    - decay: float, factor by which saliency is reduced.
    """
    radius = int(radius)

    for point in points:
        point_x = int(point[0])
        point_y = int(point[1])

        for y in range(max(0, point_y - radius), min(saliency_map.shape[0], point_y + radius + 1)):
            for x in range(max(0, point_x - radius), min(saliency_map.shape[1], point_x + radius + 1)):
                distance = np.sqrt((point_x - x) ** 2 + (point_y - y) ** 2)
                if distance <= radius:
                    saliency_map[y, x] *= (1 - decay * (1 - distance / radius))
    # Normalizing the saliency map to keep values within a reasonable range
    saliency_map = np.clip(saliency_map, 0, 1)
    

def apply_equatorial_bias(saliency_map):
    """
    Apply an equatorial bias to the saliency map to increase the saliency of points near the horizontal center.

    Parameters:
    - saliency_map: np.array, the saliency map.
    """
    rows, cols = saliency_map.shape
    equator = rows / 2

    for y in range(rows):
        for x in range(cols):
            distance_to_equator = abs(y - equator)
            decay_factor = 1 - (distance_to_equator / equator)
            saliency_map[y, x] += config.bias_strength * decay_factor * saliency_map[y, x]

    # Normalizing the saliency map to keep values within a reasonable range
    saliency_map = np.clip(saliency_map, 0, 1)

def adjust_saliency_by_angle(saliency_map, current_point, angle):
    """
    Adjust the saliency map to gradually decrease saliency of points as they are farther from the current fixation point,
    based on an angle in degrees. The angle determines the fraction of the image width that affects the saliency reduction,
    with 360 degrees representing the full width and 45 degrees representing a quarter of the width.

    Parameters:
    - saliency_map: np.array, the saliency map.
    - current_point: tuple of int, the current fixation point (x, y).
    - angle: float, the angle in degrees determining the effective 'distance' for saliency adjustment.
    """
    rows, cols = saliency_map.shape
    # Calculate max_distance based on the angle, where 2*pi radians = full width of the image
    max_distance = (np.deg2rad(angle) / (2 * np.pi)) * cols
    max_distance=cols/10

    for y in range(rows):
        for x in range(cols):
            distance = np.sqrt((current_point[0] - x) ** 2 + (current_point[1] - y) ** 2)
            if distance <= max_distance:
                decay_factor = 1 - (distance / max_distance)
                saliency_map[y, x] *= decay_factor
            else:
                saliency_map[y, x] = 0  # Set saliency to zero beyond the calculated max distance

    # Normalizing the saliency map to keep values within a reasonable range
    saliency_map = np.clip(saliency_map, 0, 1)


def generate_video_saliency_scanpath_with_inhibition(saliency_map_folder, inhibition_radius=50, inhibition_decay=0.5, history_length=5, probabilistic_importance_factor=1.0):
    """
    Generates a scanpath for each frame in a sequence of saliency maps, applying inhibition of return to recently selected points across the video frames.

    Parameters:
    - saliency_map_folder: str, Path to the folder containing saliency map images (grayscale).
    - inhibition_radius: int, Radius around selected points where saliency will be reduced.
    - inhibition_decay: float, Factor by which saliency is reduced within the inhibition radius.
    - history_length: int, Number of recent points to consider for inhibition across frames.
    - probabilistic_importance_factor: float, factor to adjust the importance given to higher saliency values.

    Returns:
    - A list of lists, each inner list represents the sequence of normalized coordinates for each frame in the video.
    """

    saliency_maps = sorted([f for f in os.listdir(saliency_map_folder) if f.endswith('.png') or f.endswith('.jpg')])
    video_scanpaths = []
    recent_points = []  # Store recent points for inhibition across frames

    for map_file in saliency_maps:
        map_path = os.path.join(saliency_map_folder, map_file)
        saliency_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if saliency_map is None:
            raise ValueError(f"Saliency map {map_path} could not be loaded.")

        if recent_points:
            modified_saliency_map = saliency_map.copy().astype(np.float32)
            apply_inhibition(modified_saliency_map, recent_points, inhibition_radius, inhibition_decay)
            if config.fixation_distance:
                adjust_saliency_by_angle(modified_saliency_map, recent_points[-1], config.fixation_angle)
        else:
            modified_saliency_map = saliency_map.astype(np.float32)

        if config.equator_bias:
            apply_equatorial_bias(modified_saliency_map)


        frame_scanpath = generate_image_probabilistic_saliency_scanpath(map_path, probabilistic_importance_factor=probabilistic_importance_factor)

        # Update recent points list for global inhibition
        recent_points.append((frame_scanpath[0] * saliency_map.shape[1], frame_scanpath[1] * saliency_map.shape[0]))
        if len(recent_points) > history_length:
            recent_points.pop(0)  # Remove the oldest point

        video_scanpaths.append(frame_scanpath)

    return video_scanpaths
