from time import time
import multiprocessing as mp
import matplotlib
matplotlib.use('pdf')
import sys
import os
import logging
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
from floortrans.loaders.augmentations import (RandomCropToSizeTorch,
                                              ResizePaddedTorch,
                                              Compose,
                                              DictToTensor,
                                              ColorJitterTorch,
                                              RandomRotations)
from torchvision.transforms import RandomChoice
from torch.utils import data
from torch.nn.functional import softmax
from tqdm import tqdm

from floortrans.loaders import FloorplanSVG
from floortrans.models import get_model
from floortrans.losses import UncertaintyLoss
from floortrans.metrics import get_px_acc, runningScore
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

aug = Compose([RandomCropToSizeTorch(data_format='dict', size=(256, 256)),
                       RandomRotations(format='cubi'),
                       DictToTensor(),
                       ColorJitterTorch()])

train_set = FloorplanSVG('data/cubicasa5k/', 'train.txt', format='lmdb',
                             augmentations=aug)
val_set = FloorplanSVG('data/cubicasa5k/', 'val.txt', format='lmdb',
                           augmentations=DictToTensor())

for num_workers in range(0, mp.cpu_count(), 2):  
    train_loader = data.DataLoader(train_set,shuffle=True,num_workers=num_workers,batch_size=26,pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, samples in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))