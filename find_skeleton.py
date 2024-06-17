import argparse
import cv2
import numpy as np
import scipy
import torch
from torch.utils import data
from floortrans.loaders import FloorplanSVG
from floortrans.loaders.augmentations import (DictToTensor)
import matplotlib.pyplot as plt
from os.path import exists
from skimage.morphology import skeletonize
from time import time
from skimage.segmentation import flood_fill




def find_skeleton(args):
    gt_floor_plan, gt_mask = ground_truth_bw(args)
    pr_floor_plan = predicted_bw(args)

    try:
        skeleton = np.load('skeleton.npy')
    except FileNotFoundError:
        start_time = time()
        skeleton = skeletonize(gt_floor_plan).astype(np.uint8)
        skeleton = np.multiply(skeleton,gt_mask, dtype=np.uint8)
        print(f"Skeletonization took: {time() - start_time:.4f} seconds")
        np.save('skeleton.npy',skeleton)
            
    adjacency_matrix = image_to_adjacency_matrix_opt(skeleton)
    # adjacency_matrix = make_graph(skeleton)

    print(adjacency_matrix.shape)
    visualize_heatmap(adjacency_matrix)

    if args.visualize:
        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(skeleton, kernel, iterations=2).astype(np.uint8)
        overlayed = gt_floor_plan - dilated_image
        cv2.imwrite('skeleton.png', skeleton*255)
        cv2.imwrite('dilated_skeleton.png', dilated_image*255)
        cv2.imwrite('overlayed.png', overlayed*255)

def image_to_adjacency_matrix_opt(image_array, nodata = None):
    image_flat = image_array.flatten()

    height, width = image_array.shape
    N = height * width
    image_has_data = image_flat != nodata
    index_dtype = np.int32 if N < 2 ** 31 else np.int64
    adjacents = np.array([
        -width - 1, -width, -width + 1,
        -1,                 1,
        width - 1,  width,  width + 1 
    ], dtype=index_dtype)
    row_idx, col_idx = np.meshgrid(
        np.arange(1, height - 1, dtype=index_dtype),
        np.arange(1, width - 1, dtype=index_dtype),
        indexing='ij'
    )
    row_idx = row_idx.reshape(-1)
    col_idx = col_idx.reshape(-1)
    pixel_idx = row_idx * width + col_idx
    pixels_with_data = image_has_data[pixel_idx]
    pixel_idx = pixel_idx[pixels_with_data]
    neighbors = pixel_idx.reshape(-1, 1) + adjacents.reshape(1, -1)
    neighbors_with_data = image_has_data[neighbors]
    row = np.repeat(pixel_idx, repeats=neighbors_with_data.sum(axis=1))
    col = neighbors[neighbors_with_data]
    data = np.ones(len(row), dtype='uint8')
    adjacency_matrix = scipy.sparse.coo_matrix((data, (row, col)), dtype=int, shape=(N, N))
    return adjacency_matrix.tocsr()

def visualize_heatmap(adjacency_matrix):
  """
  Visualizes the adjacency matrix as a heatmap.
  """

  plt.imshow(adjacency_matrix, cmap='Greys')  # Use 'Greys' for clearer visualization
  plt.colorbar(label='Edge Weight')  # Optional colorbar for edge weights (all 1 in this case)
  plt.title('Adjacency Matrix Heatmap')
  plt.xlabel('Source Node')
  plt.ylabel('Target Node')
  plt.grid(False)  # Remove grid lines for better visualization
  plt.show()


def make_graph(skeleton_image):
    adjacency_matrix = create_adjacency_matrix(skeleton_image)
    return adjacency_matrix

def create_adjacency_matrix(skeleton_image):
  """
  Creates an adjacency matrix from a binary image representing a skeleton.

  Args:
      skeleton_image: A NumPy array representing the binary skeleton image.

  Returns:
      A NumPy array representing the adjacency matrix of the graph.
  """

  # Get image dimensions
  height, width = skeleton_image.shape

  # Initialize adjacency matrix with zeros
  adjacency_matrix = np.zeros((height, width), dtype=np.uint8)

  # Iterate through each pixel
  for row in range(height):
    for col in range(width):
      if skeleton_image[row, col] == 1:  # Check if pixel is part of the skeleton
        # Check 8-neighborhood for connections
        for neighbor_row in range(row-1, row+2):
          for neighbor_col in range(col-1, col+2):
            # Check for valid neighbor coordinates within image bounds
            if 0 <= neighbor_row < height and 0 <= neighbor_col < width:
              if skeleton_image[neighbor_row, neighbor_col] == 1 and (neighbor_row, neighbor_col) != (row, col):
                # Mark connection in adjacency matrix
                adjacency_matrix[row, col] = 1
                adjacency_matrix[neighbor_row, neighbor_col] = 1  # Undirected graph (symmetric)
                break  # Stop checking neighbors if connection found

  return adjacency_matrix



def ground_truth_bw(args):
    val_set = FloorplanSVG(args.data_path, 'val.txt', format='lmdb',
                            augmentations=DictToTensor(), lmdb_folder=args.lmdb_path,
                              len_divisor=args.len_divisor)

    # samples_val = val_set[0]
    samples_val = val_set[5]
    # samples_val = val_set[9]

    with torch.no_grad():
        labels_val = samples_val['label'].cuda(non_blocking=True)
        rooms, icons = labels_val.squeeze().cpu().data.numpy()[-2:]

    walls = np.array(np.invert(rooms == 1), dtype=int)

    mask = flood_fill(walls, (0,0), 0)
    mask = np.array(np.invert(walls != mask), dtype=np.uint8)

    doors = np.array(icons == 2, dtype=int)
    floor_plan = walls + doors 
    if args.visualize:
        cv2.imwrite('mask.png', mask*255)
        cv2.imwrite('walls.png', walls*255)
        cv2.imwrite('doors.png', doors*255)
        cv2.imwrite('floor_plan.png', floor_plan*255)
    return floor_plan, mask

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
    parser.add_argument('--visualize', nargs='?', type=bool, default=True, const=True,
                    help='Save .png images that visualizes the process.')
    args = parser.parse_args()

    find_skeleton(args)
