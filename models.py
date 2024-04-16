import Modules
import torch.nn as nn
import torch
import cv2
import scanpath_generator as sg
import gaussian_map_generator as gmg
import numpy as np

class SST_Sal(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=36, output_dim=1):
        super(SST_Sal, self).__init__()

        self.encoder = Modules.SpherConvLSTM_EncoderCell(input_dim, hidden_dim)
        self.decoder = Modules.SpherConvLSTM_DecoderCell(hidden_dim, output_dim)


    def forward(self, x):

        b, _, _, h, w = x.size()
        state_e = self.encoder.init_hidden(b, (h, w))
        state_d = self.decoder.init_hidden(b, (h//2, w//2))

        outputs = []

        for t in range(x.shape[1]):
            out, state_e = self.encoder(x[:, t, :, :, :], state_e)
            out, state_d = self.decoder(out, state_d)
            outputs.append(out)
        return torch.stack(outputs, dim=1)

class SST_Sal_scanpath(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=36, output_dim=1):
        super(SST_Sal_scanpath, self).__init__()

        self.encoder = Modules.SpherConvLSTM_EncoderCell(input_dim, hidden_dim)
        self.decoder = Modules.SpherConvLSTM_DecoderCell(hidden_dim, output_dim)


    def forward(self, x, coordconv_matrix, ini_gaussian_map, device):

        b, _, _, h, w = x.size()
        state_e = self.encoder.init_hidden(b, (h, w))
        state_d = self.decoder.init_hidden(b, (h//2, w//2))

        outputs = []
        gaussian_map = ini_gaussian_map

        for t in range(x.shape[1]):
            # Obtener el frame actual
            frame_del_video = x[:, t, :, :, :]
            print('Gaussian map: ',gaussian_map.shape)

            # Concatenar el frame actual con coordconv_matrix a lo largo de la dimensión de los canales
            frame_with_coords = torch.cat((frame_del_video, coordconv_matrix, gaussian_map.to(device)), dim=1)
            print('Input: ',frame_with_coords.shape)

            # Pasar el frame concatenado a través del encoder y decoder
            out, state_e = self.encoder(frame_with_coords, state_e)
            out, state_d = self.decoder(out, state_d)
            print('Output: ',out.shape)
            out_squeezed=out.squeeze()
            print(out_squeezed.shape)
            tSPM=out_squeezed.cpu().detach().numpy()
            print('Output matrix: ',tSPM.shape)
            # cv2.imshow('tSPM :', tSPM*255)
            point = sg.generate_image_saliency_matrix_scanpath(tSPM, 1)
            gaussian_map= gmg.gaussian_map(out_squeezed.shape[0], out_squeezed.shape[1], (point[0],point[1]))
            gaussian_map = gaussian_map.astype(np.float32)
            # cv2.imshow('Mapa Gaussiano', gaussian_map)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            gaussian_map = torch.FloatTensor(gaussian_map).unsqueeze(0).unsqueeze(0)
            outputs.append(gaussian_map)
        return torch.stack(outputs, dim=1)