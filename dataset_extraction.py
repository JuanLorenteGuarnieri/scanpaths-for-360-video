import os
import cv2
from tqdm import tqdm

def frames_extraction(path_to_videos):
    """
    Extracts the frames from the videos and save them in the same folder as the original videos.
    :param path_to_videos: path to the videos
    :return:
    """
    samples_per_second = 8
    destination_folder = os.path.join(path_to_videos, 'frames')

    video_names = os.listdir(path_to_videos)
    with tqdm(range(len(video_names)), ascii=True) as pbar:
        for v_n, video_name in enumerate(video_names):

            video = cv2.VideoCapture(os.path.join(path_to_videos, video_name))
            fps = video.get(cv2.CAP_PROP_FPS)
            step = round(fps / samples_per_second)

            new_video_folder = os.path.join(destination_folder, str(v_n).zfill(3))

            if not os.path.exists(new_video_folder):
                os.makedirs(new_video_folder)

            success, frame = video.read()
            frame_id = 0
            frame_name = str(v_n).zfill(3) + '_' + str(frame_id).zfill(4) + '.png'
            cv2.imwrite(os.path.join(new_video_folder, frame_name), frame)
            frame_id += 1

            while success:
                success, frame = video.read()
                if frame_id % step == 0 and success:
                    frame_name = str(v_n).zfill(3) + '_' + str(frame_id).zfill(4) + '.png'
                    cv2.imwrite(os.path.join(new_video_folder, frame_name), frame)
                frame_id += 1
            pbar.update(1)


# Call the function to extract frames
frames_extraction("./data/videos")