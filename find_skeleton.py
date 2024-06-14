import argparse
import cv2
import numpy as np
import torch
from torch.utils import data
from floortrans.loaders import FloorplanSVG
from floortrans.loaders.augmentations import (DictToTensor)
import matplotlib.pyplot as plt
from os.path import exists
from skimage.morphology import skeletonize
from time import time



def find_skeleton(args):
    gt_floor_plan = ground_truth_bw(args)
    pr_floor_plan = predicted_bw(args)

    start_time = time()
    skeleton = skeletonize(gt_floor_plan).astype(np.uint8)
    print(f"Skeletonization took: {time() - start_time:.4f} seconds")
    # print(np.unique(skeleton))
    # cv2.imwrite('skeleton.png', skeleton*255)
    

    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(skeleton, kernel, iterations=2).astype(np.uint8)
    # cv2.imwrite('dilated_image.png', dilated_image*255)
    overlayed = gt_floor_plan - dilated_image
    cv2.imwrite('overlayed.png', overlayed*255)



def ground_truth_bw(args):
    val_set = FloorplanSVG(args.data_path, 'val.txt', format='lmdb',
                            augmentations=DictToTensor(), lmdb_folder=args.lmdb_path, len_divisor=args.len_divisor)

    samples_val = val_set[9]

    with torch.no_grad():
        labels_val = samples_val['label'].cuda(non_blocking=True)
        rooms, icons = labels_val.squeeze().cpu().data.numpy()[-2:]

    walls = np.array(np.invert(rooms == 1), dtype=int)
    doors = np.array(icons == 2, dtype=int)
    floor_plan = walls + doors 
    cv2.imwrite('walls.png', walls*255)
    cv2.imwrite('doors.png', doors*255)
    cv2.imwrite('floor_plan.png', floor_plan*255)
    return floor_plan

def predicted_bw(args):

    return None


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
