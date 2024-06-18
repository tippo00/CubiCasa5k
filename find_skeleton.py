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
            
    adjacency_matrix, coordinate_matrix = make_graph(skeleton_image=skeleton)
    print(f'Shape pre-reduction: {coordinate_matrix.shape}')
    adjacency_matrix, coordinate_matrix = reduce_deg_2_nodes(adjacency_matrix.copy(), coordinate_matrix.copy(), 40.0)
    print(f'Shape post-reduction: {coordinate_matrix.shape}')
    adjacency_matrix, coordinate_matrix = merge_by_proximity(adjacency_matrix.copy(), coordinate_matrix.copy(), 3.0)
    print(f'Shape post-reduction: {coordinate_matrix.shape}')
    print(f'Symmetric: {np.allclose(adjacency_matrix, adjacency_matrix.T)}')

    if args.visualize:
        visualize_graph(adjacency_matrix, coordinate_matrix)
        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(skeleton, kernel, iterations=2).astype(np.uint8)
        overlayed = gt_floor_plan - dilated_image
        plt.hist(np.sum(adjacency_matrix, axis=0))
        plt.savefig('hist_adjacency_matrix.png')
        cv2.imwrite('skeleton.png', skeleton*255)
        cv2.imwrite('dilated_skeleton.png', dilated_image*255)
        cv2.imwrite('overlayed.png', overlayed*255)
        cv2.imwrite('adjacency_matrix.png', adjacency_matrix*255)

# ----------------------------------------------------------------------------------------
def merge_by_proximity(adj, coords, epsilon):
  """
  This function merges nodes in a graph that are closer than a certain distance (epsilon).

  Args:
      adj: A numpy array of shape (n, n) representing the adjacency matrix of the graph.
      coords: A numpy array of shape (n, 2) representing the coordinates of n nodes.
      epsilon: A threshold for considering the distance between nodes for merging.

  Returns:
      A tuple containing a new adjacency matrix and a new coordinate matrix with merged nodes.
  """
  n = len(adj)
  visited = np.zeros(n, dtype=bool)

  # Iterate through all node pairs (avoiding duplicates)
  for i in range(n):
    for j in range(i+1, n):
      if not visited[i] and not visited[j] and np.linalg.norm(coords[i].astype(np.int32) - coords[j].astype(np.int32)) <= epsilon:
        visited[i] = True
        # Merge adjacency information (consider other merging strategies here)
        adj[i] = np.logical_or(adj[i], adj[j]).astype(np.uint8)
        adj[:,i] = np.logical_or(adj[:,i], adj[:,j]).astype(np.uint8)
        adj[j] = adj[i].copy()
        adj[:,j] = adj[:,i].copy()
        # Update coordinates for the remaining node (consider averaging or other strategies)
        # Placeholder for coordinate update strategy (e.g., averaging)
        coords[i] = (coords[i] + coords[j]) // 2  # Average coordinates

  # Remove isolated nodes (nodes with degree 0 after merging)
  new_adj = adj[~visited][:, ~visited]
  new_coords = coords[~visited]

  return new_adj, new_coords
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
def reduce_deg_2_nodes(adj, coords, epsilon):
  """
  This function reduces the number of nodes in a graph by merging highly connected nodes 
  with degree 2 (connected to only two other nodes).

  Args:
      adj: A numpy array of shape (n, n) representing the adjacency matrix of the graph.
      coords: A numpy array of shape (n, 2) representing the coordinates of n nodes.
      epsilon: A threshold for considering the distance between connected nodes for merging.

  Returns:
      A tuple containing a new adjacency matrix and a new coordinate matrix with reduced nodes.
  """
  n = len(adj)
  visited = np.zeros(n, dtype=bool)

  # Iterate until no changes are made
  while True:
    changed = False
    for i in range(n):
      if not visited[i] and adj[i].sum() == 2:  # Check for degree 2 node
        neighbors = np.where(adj[i])[0]  # Find connected neighbors
        if len(neighbors) == 2:
          j, k = neighbors
          # Check if distance between neighbors is less than epsilon
          if np.linalg.norm(coords[j].astype(np.int32) - coords[k].astype(np.int32)) <= epsilon:
            visited[i] = True
            adj[j, k] = 1  # Connect the remaining neighbors
            adj[k, j] = 1
            adj[i, :] = adj[:, i] = 0  # Isolate and remove the merged node from connections
            changed = True
            break

    if not changed:
      break

  # Remove isolated nodes (nodes with degree 0 after merging)
  new_adj = adj[~visited][:, ~visited]
  new_coords = coords[~visited]

  return new_adj, new_coords
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
def visualize_graph(adj, coords):
  """
  This function visualizes a graph represented by an adjacency matrix and a coordinate matrix.

  Args:
      adj: A numpy array of shape (n, n) representing the adjacency matrix of the graph.
      coords: A numpy array of shape (n, 2) representing the coordinates of n nodes.
  """
  plt.figure(figsize=(8, 6))

  # Draw edges
  for i in range(len(adj)):
    for j in range(i, len(adj)):
      if adj[i, j]:
        plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], 'b-o', alpha=0.7)

  # Draw nodes
  plt.scatter(coords[:, 0], coords[:, 1], marker='o', color='black', s=50)

  # Add labels for nodes (optional)
  # for i, coord in enumerate(coords):
  #   plt.annotate(i, (coord[0], coord[1]), textcoords="offset points", xytext=(0, 10), ha='center')

  plt.title("Graph Visualization")
  plt.xlabel("X-axis")
  plt.ylabel("Y-axis")
  plt.axis('off')
  plt.show()

# ----------------------------------------------------------------------------------------

def make_graph(skeleton_image):
    coordinate_matrix = create_coordinate_matrix(skeleton_image)
    adjacency_matrix = create_adjacency_matrix(skeleton_image, coordinate_matrix)
    return adjacency_matrix, coordinate_matrix

def create_coordinate_matrix(skeleton_image):
    # Get image dimensions
    height, width = skeleton_image.shape
    # Get total number of nodes
    n_nodes = np.sum(skeleton_image)

    # Initialize coordinate matrix with zeros
    coordinate_matrix = np.zeros((n_nodes, 2), dtype=np.uint16)

    i = 0
    for row in range(height):
        for col in range(width):
            if skeleton_image[row,col] == 1:
                coordinate_matrix[i,:] = [row, col]
                i = i + 1
    return coordinate_matrix


def create_adjacency_matrix(skeleton_image, coordinate_matrix):
    """
    Creates an adjacency matrix from a binary image representing a skeleton.

    Args:
        skeleton_image: A NumPy array representing the binary skeleton image.

    Returns:
        A NumPy array representing the adjacency matrix of the graph.
    """

    # Get total number of nodes
    n_nodes = np.sum(skeleton_image)

    # Initialize adjacency matrix with zeros
    adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.uint8)

    for i in range(n_nodes):
        row, col = coordinate_matrix[i, :]
        for neighbor_row in range(row-1, row+2):
            for neighbor_col in range(col-1, col+2):
                # Check for valid neighbor coordinates within image bounds
                if 0 <= neighbor_row < n_nodes and 0 <= neighbor_col < n_nodes:
                    if skeleton_image[neighbor_row, neighbor_col] == 1 and (neighbor_row, neighbor_col) != (row, col):
                        # Find node number of neighbor
                        j = np.nonzero((coordinate_matrix[:,0] == neighbor_row) * (coordinate_matrix[:,1] == neighbor_col))
                        # Mark connection in adjacency matrix
                        adjacency_matrix[i, j] = 1
                        adjacency_matrix[j, i] = 1  # Undirected graph (symmetric)
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
