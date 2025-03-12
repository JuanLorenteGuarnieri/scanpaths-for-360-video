import os
from PIL import Image
import config

# Paths
input_folder = 'data/frames'
output_folder = 'data/frames_downscaled'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to downscale images
def downscale_image(input_path, output_path, resolution):
    with Image.open(input_path) as img:
        img = img.resize((resolution[1], resolution[0]), Image.ANTIALIAS)
        img.save(output_path)

# Iterate through the input folder and process images
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith('.png'):
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_folder)
            output_dir = os.path.join(output_folder, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file)
            downscale_image(input_path, output_path, config.resolution)