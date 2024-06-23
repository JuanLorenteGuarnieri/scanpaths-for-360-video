import os
import numpy as np
import torch
import config
from DataLoader360Video import RGB_and_OF, RGB, RGB_with_GM
from torch.utils.data import DataLoader
import cv2
import tqdm
from utils import frames_extraction
import scanpath_generator as sg
import gaussian_map_generator as gmg
import json

from utils import save_video
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def eval(test_data, model, device, result_imp_path, coordconv_matrix):
    m, n = config.resolution

    model.to(device)
    model.eval()

    with torch.no_grad():

        for x, names in tqdm.tqdm(test_data):
            scanpaths = []

            for i in range(config.n_scanpaths_inference):
                outputs = []
                scanpath = []
                tSPMs = []
                gaussian_map= gmg.gaussian_map(m, n, (0.5,0.5))
                gaussian_map = gaussian_map.astype(np.float32)
                gaussian_map = torch.FloatTensor(gaussian_map).unsqueeze(0).unsqueeze(0)

                frame_del_video = x[:, 0, :, :, :]
                frame_with_coords = torch.cat((frame_del_video.to(device), coordconv_matrix.to(device), gaussian_map.to(device)), dim=1)
                state_e, state_d = model.init(frame_with_coords)

                for t in range(x.shape[1]):  # Loop over time dimension
                    frame_del_video = x[:, t, :, :, :]

                    frame_with_coords = torch.cat((frame_del_video.to(device), coordconv_matrix.to(device), gaussian_map.to(device)), dim=1)
                    out, state_e, state_d = model(frame_with_coords, state_e, state_d)
                    out_squeezed = out.squeeze()
                    tSPM = out_squeezed.cpu().detach().numpy()
                    scaled_tSPM = ((tSPM + 1) / 2) * 255
                    point = sg.generate_image_probabilistic_saliency_scanpath(scaled_tSPM, 2)
                    gaussian_map = gmg.gaussian_map(out_squeezed.shape[0], out_squeezed.shape[1], (point[1],point[0]))
                    gaussian_map = torch.FloatTensor(gaussian_map.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                    tSPMs.append(out)
                    scanpath.append(point)
                    outputs.append(gaussian_map)

                scanpaths.append(scanpath)

            outputs = torch.stack(outputs, dim=1)  # Stack along time dimension
            tSPMs = torch.stack(tSPMs, dim=1)  # Stack along time dimension

            batch_size, Nframes, _, _ = outputs[:, :, 0, :, :].shape
            # Comprobar carpetas
            scanpath_folder = os.path.join(result_imp_path, names[0][0].split('_')[0])
            os.makedirs(scanpath_folder, exist_ok=True)
            # gauss_map_folder = os.path.join(result_imp_path, names[0][0].split('_')[0],"gaus_map")
            # os.makedirs(gauss_map_folder, exist_ok=True)
            # tSPMs_folder = os.path.join(result_imp_path, names[0][0].split('_')[0],"tSPM")
            # os.makedirs(tSPMs_folder, exist_ok=True)

            # Guardar scanpath en un archivo
            scanpath_path = os.path.join(scanpath_folder,names[0][0].split('_')[0]+".scanpaths")
            with open(scanpath_path, 'w') as file:
                json.dump(scanpaths, file)


            # for bs in range(batch_size):
            #     for iFrame in range(4, Nframes):
            #         # Guardar mapas gaussianos en una carpeta
            #         sal = outputs[bs, iFrame, 0, :, :].cpu()
            #         sal = np.array((sal - torch.min(sal)) / (torch.max(sal) - torch.min(sal)))
            #         cv2.imwrite(os.path.join(gauss_map_folder, names[iFrame][bs] + '.png'), (sal * 255).astype(np.uint8))
            #         # Guardar tSPMs en una carpeta
            #         sal = tSPMs[bs, iFrame, 0, :, :].cpu()
            #         sal = np.array((sal - torch.min(sal)) / (torch.max(sal) - torch.min(sal)))
            #         cv2.imwrite(os.path.join(tSPMs_folder, names[iFrame][bs] + '.png'), (sal * 255).astype(np.uint8))



if __name__ == "__main__":

    # Extract video frames if hasn't been done yet
    if not os.path.exists(os.path.join(config.videos_folder, 'frames')):
        frames_extraction(config.videos_folder)

    # Obtain video names from the new folder 'frames'
    inference_frames_folder = os.path.join(config.videos_folder, 'frames')
    video_test_names = [config.name_inference_frames_folder]
    # Select the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device") 

    # Load the model
    model =  torch.load(config.inference_model, map_location=device)


    # Load the data. Use the appropiate data loader depending on the expected input data
    # test_video360_dataset = RGB_with_GM(inference_frames_folder, None, video_test_names, config.sequence_length, split='test', load_names=True, skip=0, resolution=config.resolution)
    test_video360_dataset = RGB(inference_frames_folder, None, video_test_names, config.sequence_length, split='test', load_names=True, skip=0, resolution=config.resolution)

    test_data = DataLoader(test_video360_dataset, batch_size=config.batch_size, shuffle=False)

    # Prepare the CoordConv matrix similar to the training phase
    images, labels = next(iter(test_data))
    m, n = images.size()[3:5]  # Obtener las dimensiones espaciales: altura (m) y anchura (n)

    # Crear la matriz de coordenadas Y
    coord_y = torch.arange(float(m)).unsqueeze(1).expand(m, n)
    # Crear la matriz de coordenadas X
    coord_x = torch.arange(float(n)).unsqueeze(0).expand(m, n)

    # Apilar las dos matrices para crear la matriz de coordenadas convolucionales
    coordconv_matrix = torch.stack((coord_x, coord_y), dim=0)
    # torch.tensor(data_numpy)
    # coordconv_matrix = coordconv_matrix.astype(np.float32)

    coordconv_matrix = torch.FloatTensor(coordconv_matrix).unsqueeze(0)

    eval(test_data, model, device, config.results_dir, coordconv_matrix)
    # Save video with the results

    # for video_name in video_test_names:
    #     save_video(os.path.join(inference_frames_folder, video_name), 
    #             os.path.join(config.results_dir, video_name),
    #             None,
    #             'SST-Sal_pred_' + video_name +'.avi')
