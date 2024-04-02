import numpy as np

def spherical_equirectangular_distance(x, y, center_x, center_y, width, height):
    """
    Calculates distance in an equirectangular space considering spherical geometry.
    """
    # Convert pixel coordinates to spherical coordinates (longitude and latitude)
    longitude = (x / width) * 2 * np.pi - np.pi  # From -pi to pi
    latitude = (y / height) * np.pi - (np.pi / 2)  # From -pi/2 to pi/2
    
    center_longitude = (center_x / width) * 2 * np.pi - np.pi
    center_latitude = (center_y / height) * np.pi - (np.pi / 2)
    
    # Calculate angular difference in longitude and latitude
    delta_longitude = np.minimum(np.abs(longitude - center_longitude), 2 * np.pi - np.abs(longitude - center_longitude))
    delta_latitude = np.abs(latitude - center_latitude)
    
    # Use spherical distance formula (law of cosines for spherical trigonometry)
    # Consider a sphere with radius 1 for simplification, as we're interested in relative distance
    angular_distance = np.arccos(np.sin(center_latitude) * np.sin(latitude) +
                                 np.cos(center_latitude) * np.cos(latitude) * np.cos(delta_longitude))
    
    # Equirectangular distance will be proportional to the angular distance
    distance = height / np.pi * angular_distance  # Proportional to the height of the image
    
    return distance

def equirectangular_distance(x, y, center_x, center_y, width, height):
    """
    Calculates distance in an equirectangular space considering planar geometry.
    """
    # Difference in Y coordinates
    delta_y = np.abs(y - center_y)
    
    # Minimum difference in X axis considering the equirectangular projection wrap-around
    delta_x_min = np.minimum(np.abs(x - center_x), width - np.abs(x - center_x))
    
    # Calculate the real distance considering equirectangular distortion
    distance = np.sqrt(delta_x_min**2 + delta_y**2)
    
    return distance

def gaussian_map(height, width, point, radius, equirectangular=True):
    """
    Creates a black image with specified dimensions and draws a white point with a Gaussian gradient around it.
    """
    image = np.zeros((height, width), dtype=np.float32)
    
    # Calculate real coordinates of the circle's center
    center_x = int(point[0] * width)
    center_y = int(point[1] * height)
    
    # Create coordinate matrices
    y, x = np.ogrid[:height, :width]
    
    # Calculate distance matrix to the center
    if equirectangular:
        distance_to_center = spherical_equirectangular_distance(x, y, center_x, center_y, width, height)
    else:
        distance_to_center = equirectangular_distance(x, y, center_x, center_y, width, height)
    
    # Calculate intensity based on a Gaussian function
    sigma = radius / 3  # Set sigma so the gradient falls to 0 around the circle's edge
    intensity = np.exp(-(distance_to_center**2 / (2 * sigma**2)))
    intensity[distance_to_center > radius] = 0  # Ensure intensity is 0 outside the radius
    
    # Normalize intensity to range [0, 1]
    image = intensity / np.max(intensity)
    
    return image