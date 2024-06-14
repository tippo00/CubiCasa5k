import argparse
import cv2
import numpy as np
import torch
from torch.utils import data
from floortrans.loaders import FloorplanSVG
from floortrans.loaders.augmentations import (DictToTensor)
import matplotlib.pyplot as plt
from os.path import exists


def find_skeleton(args):
    # if not exists('floor-plan.png'):
    # floor_plan = cv2.imread('floor-plan.png')
    # doors = cv2.imread('floor-plan_doors.png')
    walls, doors = ground_truth_bw(args)
    floor_plan = walls + doors 
    cv2.imwrite('floor_plan.png', floor_plan*255)



def ground_truth_bw(args):
    val_set = FloorplanSVG(args.data_path, 'val.txt', format='lmdb',
                            augmentations=DictToTensor(), lmdb_folder=args.lmdb_path, len_divisor=args.len_divisor)

    num_workers = 0

    # valloader = data.DataLoader(val_set, batch_size=1,
    #                                 num_workers=num_workers, pin_memory=True)
    # next(iter(valloader))
    # next(iter(valloader))
    # samples_val = next(iter(valloader))

    samples_val = val_set[9]

    with torch.no_grad():
        labels_val = samples_val['label'].cuda(non_blocking=True)
        rooms, icons = labels_val.squeeze().cpu().data.numpy()[-2:]

    walls = np.array(np.invert(rooms == 1), dtype=int)
    doors = np.array(icons == 2, dtype=int)
    cv2.imwrite('walls.png', walls*255)
    cv2.imwrite('doors.png', doors*255)
    return walls, doors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', nargs='?', type=str, default='data/cubicasa5k/',
                            help='Path to data directory')
    parser.add_argument('--lmdb-path', nargs='?', type=str, default='cubi_lmdb_27_classes//',
                        help='Path to lmdb')
    parser.add_argument('--len-divisor', nargs='?', type=int, default=400,
                    help='Number with which to divide the size of the train and val dataset.')
    args = parser.parse_args()

    find_skeleton(args)
