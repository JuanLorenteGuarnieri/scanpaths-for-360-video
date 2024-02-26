import config
import scanpath_visualizer as sv
import scanpath_generator as sg
import shutil
import os
import cv2

def process_video_scanpath():
    saliency_map_folder = os.path.join(config.folder_path, 'saliency')
    original_folder = os.path.join(config.folder_path, 'original')
    output_folder = os.path.join(config.folder_path, 'output')
    frames_folder = os.path.join(output_folder, 'frames')  # Subfolder for frames
    
    # If output_folder already exists, delete it and create a new one
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(frames_folder, exist_ok=True)  # Also create the subfolder for frames

    # Generate scanpaths for each frame based on saliency maps
    if config.scanpath_generator_type == 'random':
        video_scanpath = sg.generate_video_random_scanpath(original_folder)
    elif config.scanpath_generator_type == 'max_saliency':
        video_scanpath = sg.generate_video_saliency_scanpath(saliency_map_folder)
    elif config.scanpath_generator_type == 'percentile_saliency':
        video_scanpath = sg.generate_video_random_saliency_scanpath(saliency_map_folder, config.percentile)
    elif config.scanpath_generator_type == 'probabilistic_saliency':
        video_scanpath = sg.generate_video_probabilistic_saliency_scanpath(saliency_map_folder, config.probabilistic_importance)
    elif config.scanpath_generator_type == 'inhibition_saliency':
        video_scanpath = sg.generate_video_saliency_scanpath_with_inhibition(saliency_map_folder, config.inhibition_radius, config.inhibition_decay, config.inhibition_history_length, config.probabilistic_importance)
    else:
        raise ValueError("Unsupported scanpath generator type specified in config.py")

    # Visualize the scanpaths on the original images, saving them to frames_folder
    if config.overlay_saliency and config.scanpath_generator_type != 'random':
      sv.visualize_video_scanpath(original_folder, video_scanpath, path_to_save=frames_folder, overlay_folder=saliency_map_folder)
    else:
      sv.visualize_video_scanpath(original_folder, video_scanpath, path_to_save=frames_folder)

    # Generate video from the frames
    frame_files = sorted(os.listdir(frames_folder))  # Assuming frames are named in a sortable manner
    if frame_files:
        # Assuming all frames have the same size
        first_frame_path = os.path.join(frames_folder, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        height, width, layers = first_frame.shape

        extension_type = config.scanpath_generator_type
        if config.scanpath_generator_type == 'percentile_saliency':
            extension_type = extension_type + '_' + str(config.percentile)
        elif config.scanpath_generator_type == 'probabilistic_saliency':
            extension_type = extension_type + '_' + str(config.probabilistic_importance)
        elif config.scanpath_generator_type == 'inhibition_saliency':
            extension_type = extension_type + '_R' + str(config.inhibition_radius)+ '_D' + str(config.inhibition_decay) + '_L' + str(config.inhibition_history_length)

        # Extract folder name for the video file name
        video_name = os.path.basename(config.folder_path+'_'+ extension_type) + '.avi'  # Or use .mp4
        video_path = os.path.join(output_folder, video_name)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Adjust codec as needed
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))

        for frame_file in frame_files:
            frame_path = os.path.join(frames_folder, frame_file)
            frame = cv2.imread(frame_path)
            out.write(frame)  # Write the frame to the video

        out.release()  # Release the VideoWriter object

    print("Video exported.")


if __name__ == "__main__":
    process_video_scanpath()