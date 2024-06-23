import Modules
import torch.nn as nn
import torch
import cv2
import scanpath_generator as sg
import gaussian_map_generator as gmg
import numpy as np

class SST_Sal_scanpath(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=36, output_dim=1):
        super(SST_Sal_scanpath, self).__init__()

        self.encoder = Modules.SpherConvLSTM_EncoderCell(input_dim, hidden_dim)
        self.decoder = Modules.SpherConvLSTM_DecoderCell(hidden_dim, output_dim)


    def init(self, x):

        b, _, h, w = x.size()
        state_e = self.encoder.init_hidden(b, (h, w))
        state_d = self.decoder.init_hidden(b, (h//2, w//2))
        return state_e, state_d


    def forward(self, x, state_e, state_d):

        out, state_e = self.encoder(x, state_e)
        out, state_d = self.decoder(out, state_d)
        return out, state_e, state_d