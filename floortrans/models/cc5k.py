import torch
import torch.nn as nn
import torch.nn.functional as F
from .hg_furukawa_original import *


class cc5k(hg_furukawa_original):
    def init_weights(self):
        # # Pre-trained network weights from CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis

        checkpoint = torch.load('floortrans/models/cc5k_weights.pkl')
        self.load_state_dict(checkpoint['model_state'])
