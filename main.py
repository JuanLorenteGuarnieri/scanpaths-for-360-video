import config
import scanpath_visualizer as sv
import scanpath_generator as sg
import metrics as metrics
import utils as utils
import numpy as np
import shutil
import os
import cv2
import json
from itertools import combinations

scanpaths = []


def analyzer():
    # Check if the file exists
    if not os.path.exists("./output_scanpaths/"+config.a_name + config.a_parameters+ ".scanpaths"):
        print(f"File {config.a_name + config.a_parameters+ '.scanpaths'} not found.")
        return

    # Load the scanpaths from the file
    with open("./output_scanpaths/"+config.a_name + config.a_parameters + ".scanpaths", 'r') as file:
        scanpaths = json.load(file)

    scanpaths_scaled = [[[int(x * 100), int(y * 100)] for x, y in scanpath] for scanpath in scanpaths]

    dtw_scores, det_scores, rec_scores, lev_scores, tde_scores, eye_scores, euc_scores, frech_scores = [], [], [], [], [], [], [], []

    # metrics.SIM(saliency_map, saliency_map_gt) # predicted saliency map VS ground truth saliency map
    # metrics.mannan_distance(sp1,sp2, 100, 100, PR=None, QR=None) # PR y QR son scanpaths random

    # metrics.TDE(sp1, sp2, k=3, distance_mode='Mean')
    # metrics.eyenalysis(sp1, sp2)
    # metrics.euclidean_distance(sp1, sp2)
    # metrics.frechet_distance(sp1, sp2)
    
    print(metrics.eyenalysis(scanpaths_scaled[0], scanpaths_scaled[1]))
    return

    for sp1, sp2 in combinations(scanpaths_scaled, 2):
        rec_score = [0] * 8  # Reserva espacio para 4 métricas
        rec_score[0] = metrics.DTW(sp1, sp2)
        rec_score[1] = metrics.DET(sp1, sp2, 99)
        rec_score[2] = metrics.REC(sp1, sp2, 10)
        rec_score[3] = metrics.levenshtein_distance(np.array(sp1), np.array(sp2), 100, 100)
        rec_score[4] = metrics.TDE(sp1, sp2, k=3, distance_mode='Mean')
        rec_score[5] = 0 #metrics.eyenalysis(sp1, sp2)
        rec_score[6] = metrics.euclidean_distance(sp1, sp2)
        rec_score[7] = metrics.frechet_distance(sp1, sp2)

        dtw_scores.append(rec_score[0])
        det_scores.append(rec_score[1])
        rec_scores.append(rec_score[2])
        lev_scores.append(rec_score[3])
        tde_scores.append(rec_score[4])
        eye_scores.append(rec_score[5])
        euc_scores.append(rec_score[6])
        frech_scores.append(rec_score[7])

    average_dtw = np.mean(dtw_scores) if dtw_scores else 0
    average_det = np.mean(det_scores) if det_scores else 0
    average_rec = np.mean(rec_scores) if rec_scores else 0
    average_lev = np.mean(lev_scores) if lev_scores else 0
    average_tde = np.mean(tde_scores) if tde_scores else 0
    average_eye = np.mean(eye_scores) if eye_scores else 0
    average_euc = np.mean(euc_scores) if euc_scores else 0
    average_frech = np.mean(frech_scores) if frech_scores else 0

    print(f"Average DTW: {average_dtw}\n Average DET: {average_det}\n Average REC: {average_rec}\n Average Levenshtein: {average_lev}\n Average TDE: {average_tde}\n Average eyenalysis: {average_eye}\n Average Euclidean: {average_euc}\n Average Frechet: {average_frech}")

def visualizer():
    # Check if the file exists
    if not os.path.exists("./output_scanpaths/"+config.v_name + config.v_parameters+ ".scanpaths"):
        print(f"File {config.v_name + config.v_parameters+ '.scanpaths'} not found.")
        return

    # Load the scanpaths from the file
    with open("./output_scanpaths/"+config.v_name + config.v_parameters + ".scanpaths", 'r') as file:
        scanpaths = json.load(file)

    # Ensure config.i_scanpath is a valid index
    i_scanpath = min(config.i_scanpath-1, len(scanpaths) - 1)

    output_folder = "./output_scanpaths/"+config.v_name + config.v_parameters+"_n"+str(i_scanpath)
    frames_folder = os.path.join(output_folder, 'frames')  # Subfolder for frames

    # If output_folder already exists, delete it and create a new one
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    else:
        os.makedirs(output_folder, exist_ok=True)
    os.makedirs(frames_folder, exist_ok=True)  # Also create the subfolder for frames

    # Select the scanpath
    selected_scanpath = scanpaths[i_scanpath]

    if os.path.exists("./data/"+config.v_name + "/original/"):
        original_video_path = "./data/"+config.v_name + "/original/"
    else:
        original_video_path = None

    if os.path.exists("./data/"+config.v_name + "/saliency/"):
        saliency_video_path = "./data/"+config.v_name + "/saliency/"
    else:
        saliency_video_path = None

    # Visualize the selected scanpath
    if config.v_type == "preview":
        sv.visualize_video_scanpath(original_video_path, scanpaths, path_to_save=frames_folder, overlay_folder=saliency_video_path, history_length=config.v_history_length, generated=False)
    elif config.v_type == "multi":
        fov_vert_hor = None
        utils.plot_all_viewports(scanpaths, fov_vert_hor, frames_folder, config.v_name)
    elif config.v_type == "thumbnail":
        utils.plot_thumbnail(selected_scanpath, frames_folder, config.v_name)

    # Generate video from the frames
    frame_files = sorted(os.listdir(frames_folder))  # Assuming frames are named in a sortable manner
    if frame_files:
        # Assuming all frames have the same size
        first_frame_path = os.path.join(frames_folder, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        height, width, layers = first_frame.shape

        # Extract folder name for the video file name
        if config.v_type == "preview":
            video_name = config.v_name + config.v_parameters+"_n"+str(i_scanpath) + '_preview.avi'
        elif config.v_type == "multi":
            video_name = config.v_name + config.v_parameters+'_multi.avi'
        elif config.v_type == "thumbnail":
            video_name = config.v_name + config.v_parameters+"_n"+str(i_scanpath) + '_thumbnail.avi'
        video_path = os.path.join(output_folder, video_name)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Adjust codec as needed
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))

        for frame_file in frame_files:
            frame_path = os.path.join(frames_folder, frame_file)
            frame = cv2.imread(frame_path)
            out.write(frame)  # Write the frame to the video

        out.release()  # Release the VideoWriter object

    print(f"Scanpath visualization complete")

def generator():
    saliency_map_folder = os.path.join(config.folder_path, 'saliency')
    original_folder = os.path.join(config.folder_path, 'original')
    output_folder = os.path.join(config.folder_path, 'output')
    frames_folder = os.path.join(output_folder, 'frames')  # Subfolder for frames

    # If output_folder already exists, delete it and create a new one
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(frames_folder, exist_ok=True)  # Also create the subfolder for frames

    for i in range(config.n_scanpaths):
        print(f"Processing scanpath {i+1}/{config.n_scanpaths}...")

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
        if not config.visualize:
            scanpaths.append(video_scanpath)
        else:
            if config.overlay_saliency and config.scanpath_generator_type != 'random':
                sv.visualize_video_scanpath(original_folder, video_scanpath, path_to_save=frames_folder, overlay_folder=saliency_map_folder, history_length=config.g_history_length)
            else:
                sv.visualize_video_scanpath(original_folder, video_scanpath, path_to_save=frames_folder, history_length=config.g_history_length)

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
                    if config.equator_bias:
                        extension_type = extension_type + '_EB' + str(config.bias_strength)
                    if config.fixation_distance:
                        extension_type = extension_type + '_FixAngle' + str(config.fixation_angle)

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

        if i == config.n_scanpaths - 1:

            extension_type = config.scanpath_generator_type
            if config.scanpath_generator_type == 'percentile_saliency':
                extension_type = extension_type + '_' + str(config.percentile)
            elif config.scanpath_generator_type == 'probabilistic_saliency':
                extension_type = extension_type + '_' + str(config.probabilistic_importance)
            elif config.scanpath_generator_type == 'inhibition_saliency':
                extension_type = extension_type + '_R' + str(config.inhibition_radius)+ '_D' + str(config.inhibition_decay) + '_L' + str(config.inhibition_history_length)
                if config.equator_bias:
                    extension_type = extension_type + '_EB' + str(config.bias_strength)
                if config.fixation_distance:
                    extension_type = extension_type + '_FixAngle' + str(config.fixation_angle)

            # Extract folder name for the video file name
            scanpath_filename = os.path.basename(config.folder_path+"_N"+str(config.n_scanpaths)+"_"+ extension_type) + '.scanpaths'  # Or use .mp4
            scanpath_path = os.path.join("./output_scanpaths/", scanpath_filename)

            # Guardar la lista scanpaths en un archivo
            with open(scanpath_path, 'w') as file:
                json.dump(scanpaths, file)

            print(f"Scanpaths saved to {scanpath_path}")


if __name__ == "__main__":
    if config.analyze:
        analyzer()
    elif not config.generate and config.visualize:
        visualizer()
    elif config.generate:
        generator()
    elif not config.analyze and not config.generate and not config.visualize:
        print(f"No option selected")