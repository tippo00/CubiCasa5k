import argparse
import cv2
import numpy as np
import scipy
import torch
from torch.utils import data
from floortrans.models import get_model
from floortrans.loaders import FloorplanSVG
from floortrans.loaders.augmentations import (DictToTensor)
from floortrans.loaders.augmentations import RotateNTurns
from torch.nn.functional import sigmoid, softmax, interpolate
from floortrans import post_prosessing
import matplotlib.pyplot as plt
from os.path import exists
from skimage.morphology import skeletonize
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.morphology import (binary_erosion, binary_dilation, binary_closing, binary_opening,
                                area_closing, area_opening,square)
from time import time
from skimage.segmentation import flood_fill
from pybind11_rdp import rdp
from tqdm import tqdm


room_cls_dict = {
    44: ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bedroom", "Bath", "Hallway", "Railing", "Storage", "Garage", "Other rooms"],
    29: ["Background", "Outdoor", "Wall", "Room", "Railing"],
    27: ["Background", "Wall", "Room"]
}
icon_cls_dict = {
    44: ["Empty", "Window", "Door", "Closet", "Electr. Appl.", "Toilet", "Sink", "Sauna bench", "Fire Place", "Bathtub", "Chimney"],
    29: ["Empty", "Window", "Door"],
    27: ["Empty", "Window", "Door"]
}

image_folder = 'debug_images/'

def find_skeleton(args):
    gt_floor_plan, gt_mask = ground_truth_bw(args)
    pr_floor_plan, pr_mask = predicted_bw(args)

    def perform_skeletonization():
        start_time = time()
        gt_skeleton = skeletonize(gt_floor_plan).astype(np.uint8)
        pr_skeleton = skeletonize(pr_floor_plan).astype(np.uint8)
        gt_skeleton = np.multiply(gt_skeleton,gt_mask, dtype=np.uint8)
        pr_skeleton = np.multiply(pr_skeleton,pr_mask, dtype=np.uint8)
        print(f"Skeletonization took: {time() - start_time:.4f} seconds")
        np.save(f'{image_folder}gt_skeleton.npy',gt_skeleton)
        np.save(f'{image_folder}pr_skeleton.npy',pr_skeleton)
        return gt_skeleton, pr_skeleton

    if args.re_skeletonize:
        gt_skeleton, pr_skeleton = perform_skeletonization()
    else:
        try:
            gt_skeleton = np.load(f'{image_folder}gt_skeleton.npy')
            pr_skeleton = np.load(f'{image_folder}pr_skeleton.npy')
        except FileNotFoundError:
            gt_skeleton, pr_skeleton = perform_skeletonization()

            
    gt_adj_matrix, gt_coord_matrix = make_graph(skeleton_image=gt_skeleton)
    print(f'Shape pre-reduction: {gt_coord_matrix.shape}')
    
    gt_adj_matrix, gt_coord_matrix = RDP_graph(gt_adj_matrix.copy(), gt_coord_matrix.copy(), 18) # Interesting values: 0.1, 18
    gt_adj_matrix, gt_coord_matrix = merge_by_proximity(gt_adj_matrix.copy(), gt_coord_matrix.copy(), 22) # Interesting values: 3.61, 10
    print(f'Shape post-reduction: {gt_coord_matrix.shape}')
    print(f'Symmetric: {np.allclose(gt_adj_matrix, gt_adj_matrix.T)}')

    # visualize_graph_overlayed(adj_matrix, coord_matrix, gt_floor_plan,show=False)


    pr_adj_matrix, pr_coord_matrix = make_graph(skeleton_image=pr_skeleton)
    print(f'Shape pre-reduction: {pr_coord_matrix.shape}')

    pr_adj_matrix, pr_coord_matrix = RDP_graph(pr_adj_matrix.copy(), pr_coord_matrix.copy(), 18) # Interesting values: 0.1, 18
    pr_adj_matrix, pr_coord_matrix = merge_by_proximity(pr_adj_matrix.copy(), pr_coord_matrix.copy(), 22) # Interesting values: 3.61, 10
    print(f'Shape post-reduction: {pr_coord_matrix.shape}')
    print(f'Symmetric: {np.allclose(pr_adj_matrix, pr_adj_matrix.T)}')

    visualize_graph_overlayed(pr_adj_matrix, pr_coord_matrix, pr_floor_plan,'pr_graph_overlayed',show=False)
    # visualize_both_graphs_overlayed((gt_adj_matrix, gt_coord_matrix, gt_floor_plan),
    #                                  (pr_adj_matrix, pr_coord_matrix, pr_floor_plan), show=True)

    if args.visualize:
        visualize_graph(gt_adj_matrix, gt_coord_matrix,show=False,labels=False)
        visualize_graph(pr_adj_matrix, pr_coord_matrix,show=False,labels=False)
        kernel = np.ones((3, 3), np.uint8)
        gt_dilated_image = cv2.dilate(gt_skeleton, kernel, iterations=2).astype(np.uint8)
        pr_dilated_image = cv2.dilate(pr_skeleton, kernel, iterations=2).astype(np.uint8)
        gt_overlayed = gt_floor_plan - gt_skeleton
        pr_overlayed = pr_floor_plan - pr_skeleton
        plt.clf()
        plt.hist(np.sum(gt_adj_matrix, axis=0))
        plt.savefig(f'{image_folder}hist_adjacency_matrix.png')
        cv2.imwrite(f'{image_folder}gt_skeleton.png', gt_skeleton*255)
        cv2.imwrite(f'{image_folder}pr_skeleton.png', pr_skeleton*255)
        cv2.imwrite(f'{image_folder}gt_dilated_skeleton.png', gt_dilated_image*255)
        cv2.imwrite(f'{image_folder}pr_dilated_skeleton.png', pr_dilated_image*255)
        cv2.imwrite(f'{image_folder}gt_overlayed.png', gt_overlayed*255)
        cv2.imwrite(f'{image_folder}pr_overlayed.png', pr_overlayed*255)
        cv2.imwrite(f'{image_folder}gt_adjacency_matrix.png', gt_adj_matrix*255)
        cv2.imwrite(f'{image_folder}pr_adjacency_matrix.png', pr_adj_matrix*255)

# ----------------------------------------------------------------------------------------
def RDP_graph(adj, coords, epsilon):
    new_adj = adj.copy()
    new_coords = coords.copy()

    # Filter out all nodes that are degree 2
    mask = np.sum(adj, axis=0) == 2
    remove = np.zeros_like(mask)

    # Find connected components only among the degree 2 nodes
    n_comp, components = scipy.sparse.csgraph.connected_components(adj[mask][:, mask], directed=False)

    # Iterate through all connected components
    for i in range(n_comp):
        # Prepping data
        loop_mask = np.zeros_like(mask)
        loop_mask[mask] = components==i
        if np.sum(loop_mask) < 2:
           continue

        debug = False
        if debug:
            cv2.imwrite(f'{image_folder}pre_sort_adj.png', adj[np.ix_(loop_mask,loop_mask)]*255)
            visualize_graph(adj[np.ix_(loop_mask,loop_mask)],coords[loop_mask],'pre_sort_visualization',False,True)

        idx_map = sort_map(adj, coords, loop_mask)

        if debug:
            visualize_graph(adj[np.ix_(idx_map,idx_map)],coords[idx_map],'pre_rdp_visualization',False,False)
            cv2.imwrite(f'{image_folder}pre_rdp_adj.png', adj[np.ix_(idx_map,idx_map)]*255)
        
        # Doing RDP
        return_mask = rdp(coords[idx_map],epsilon=epsilon,return_mask=True).astype(bool)
        keep_map = idx_map[return_mask]
        lose_map = idx_map[~return_mask]
        remove[lose_map] = True

        # Fixing adjacency
        n = len(keep_map)
        new_adj[np.ix_(keep_map,keep_map)] = np.diag(np.ones(n-1),1) + \
                                                        np.diag(np.ones(n-1),-1)
        new_adj[lose_map,:] = 0
        new_adj[:,lose_map] = 0

        if debug:
            visualize_graph(new_adj[keep_map][:, keep_map],new_coords[keep_map],'post_rdp_visualization',False,True)
            plt.close()
            cv2.imwrite(f'{image_folder}post_rdp_adj.png', new_adj[keep_map][:, keep_map]*255)

    # Remove isolated nodes (nodes with degree 0 after merging)
    new_adj = new_adj[np.ix_(~remove,~remove)]
    new_coords = new_coords[~remove]

    return new_adj, new_coords
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
def sort_map(adj,coords, mask):
    n = np.sum(mask)
    map = np.zeros(n,dtype=np.uint32)
    idx_endpoints = np.nonzero(np.sum(adj[np.ix_(mask,mask)], axis=0) == 1)[0]
    map[0] = np.nonzero(mask)[0][idx_endpoints[0]]

    for i in range(n-1):
        connection_1 = np.nonzero(adj[map[i]])[0][1]
        connection_2 = np.nonzero(adj[map[i]])[0][0]
        if connection_1 not in map and connection_1 in np.nonzero(mask)[0]:
            map[i+1] = connection_1
        else:
            map[i+1] = connection_2
    return map
# ----------------------------------------------------------------------------------------
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
        adj[j,j] = 0    # Remove self-loops
        # Update coordinates for the remaining node (consider averaging or other strategies)
        coords[i] = (coords[i] + coords[j]) // 2  # Average coordinates

  # Remove isolated nodes (nodes with degree 0 after merging)
  new_adj = adj[~visited][:, ~visited]
  new_coords = coords[~visited]

  return new_adj, new_coords
# ----------------------------------------------------------------------------------------
def visualize_both_graphs_overlayed(ground_truth,predicted,show=False):
    gt_adj_matrix, gt_coord_matrix, gt_floor_plan = ground_truth
    pr_adj_matrix, pr_coord_matrix, pr_floor_plan = predicted
    filename = 'both_graphs_overlayed'

    def draw_plot(adj,coords,floorplan,ax):
        # Draw edges
        for i in range(len(adj)):
            for j in range(i, len(adj)):
                if adj[i, j]:
                    ax.plot([coords[i, 1], coords[j, 1]], [coords[i, 0], coords[j, 0]], 'b-o', alpha=0.7)

        # Draw nodes
        ax.scatter(coords[:, 1], coords[:, 0], marker='o', color='black', s=50)
        # ax.xlabel("X-axis")
        # ax.ylabel("Y-axis")
        ax.axis('off')
        ax.axis('equal')
        ax.imshow(np.invert(floorplan),cmap='Greys')
        # ax.savefig(f'{filename}.png')
        return ax
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1 = draw_plot(gt_adj_matrix,gt_coord_matrix,gt_floor_plan,ax1)
    ax2 = draw_plot(pr_adj_matrix,pr_coord_matrix,pr_floor_plan,ax2)
    ax1.set_title("Ground Truth")
    ax2.set_title("Prediction")
    fig.suptitle("Graph Visualization")
    fig.savefig(f'{image_folder}{filename}.png')
    
    if show:
        plt.show(block=True)

    return
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
def visualize_graph_overlayed(adj,coords,floorplan,filename = 'graph_overlayed',show=False):
    

    plt.figure(figsize=(12,9))
    # Draw edges
    for i in range(len(adj)):
        for j in range(i, len(adj)):
            if adj[i, j]:
                plt.plot([coords[i, 1], coords[j, 1]], [coords[i, 0], coords[j, 0]], 'b-o', alpha=0.7)

    # Draw nodes
    plt.scatter(coords[:, 1], coords[:, 0], marker='o', color='black', s=50)
    plt.title("Graph Visualization")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.axis('off')
    plt.axis('equal')
    plt.imshow(np.invert(floorplan),cmap='Greys')
    plt.savefig(f'{image_folder}{filename}.png')
    if show:
        plt.show()

    return
# ----------------------------------------------------------------------------------------
def visualize_graph(adj, coords,filename='graph_visualization',show=True,labels=False):
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
        plt.plot([coords[i, 1], coords[j, 1]], [coords[i, 0], coords[j, 0]], 'b-o', alpha=0.7)

  # Draw nodes
  plt.scatter(coords[:, 1], coords[:, 0], marker='o', color='black', s=50)

  # Add labels for nodes (optional)
  if labels:
    for i, coord in enumerate(coords):
      plt.annotate(f'{i}', (coord[1], coord[0]), textcoords="offset points", xytext=(0, 10), ha='center')

  plt.title("Graph Visualization")
  plt.xlabel("X-axis")
  plt.ylabel("Y-axis")
  plt.axis('off')
  plt.axis('equal')
  plt.gca().invert_yaxis()
  plt.savefig(f'{image_folder}{filename}.png')
  if show: 
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

    # Finding wall index
    wall_index = room_cls_dict[args.n_classes].index('Wall')

    walls = np.array(np.invert(rooms == wall_index), dtype=int)

    mask = flood_fill(walls, (0,0), 0)
    mask = np.array(np.invert(walls != mask), dtype=np.uint8)

    doors = np.array(icons == 2, dtype=int)
    floor_plan = walls + doors 
    if args.visualize:
        cv2.imwrite(f'{image_folder}gt_mask.png', mask*255)
        cv2.imwrite(f'{image_folder}gt_walls.png', walls*255)
        cv2.imwrite(f'{image_folder}gt_doors.png', doors*255)
        cv2.imwrite(f'{image_folder}gt_floor_plan.png', floor_plan*255)
    return floor_plan, mask

def predicted_bw(args):
    val_set = FloorplanSVG(args.data_path, 'val.txt', format='lmdb',
                            augmentations=DictToTensor(), lmdb_folder=args.lmdb_path,
                              len_divisor=args.len_divisor)

    data_loader = data.DataLoader(val_set, batch_size=1, num_workers=0)

    # samples_val = val_set[0]
    # samples_val = val_set[5]
    # samples_val = val_set[9]

    checkpoint = torch.load(args.weights)
    # Setup Model
    if args.arch == 'hg_furukawa_original':
        model = get_model(args.arch, 51)
    elif args.arch == 'cc5k':
        model = get_model(args.arch, 44)

    n_classes = args.n_classes
    split = {
        44: [21, 12, 11],
        29: [21, 5, 3],
        27: [21, 3, 3]
    }

    split = split[args.n_classes]
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.cuda()

    with torch.no_grad():
        for count, val in tqdm(enumerate(data_loader), total=len(data_loader), ncols=80, leave=False):
            if count != 5:
                continue
            images_val = val['image'].cuda()
            labels_val = val['label']
            height = labels_val.shape[2]
            width = labels_val.shape[3]
            img_size = (height, width)

            rotate = True
            if rotate:
                rot = RotateNTurns()
                rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
                pred_count = len(rotations)
                prediction = torch.zeros([pred_count, n_classes, height, width])
                for i, r in enumerate(rotations):
                    forward, back = r
                    # We rotate first the image
                    rot_image = rot(images_val, 'tensor', forward)
                    pred = model(rot_image)
                    # We rotate prediction back
                    pred = rot(pred, 'tensor', back)
                    # We fix heatmaps
                    pred = rot(pred, 'points', back)
                    # We make sure the size is correct
                    pred = interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
                    # We add the prediction to output
                    prediction[i] = pred[0]

                prediction = torch.mean(prediction, 0, True)
            else:
                prediction = model(images_val)
            
            heatmaps, rooms, icons = post_prosessing.split_prediction(
                prediction, img_size, split)
            
            rooms_seg = np.argmax(rooms, axis=0)
            icons_seg = np.argmax(icons, axis=0)

    # Finding wall index
    wall_index = room_cls_dict[args.n_classes].index('Wall')

    walls = np.array(rooms_seg == wall_index, dtype=bool)
    
    walls = binary_erosion(walls,square(2))
    walls = binary_closing(walls,square(9))

    walls = np.array(np.invert(walls), dtype=int)

    mask = flood_fill(walls, (0,0), 0)
    mask = np.array(np.invert(walls != mask), dtype=np.uint8)

    doors = np.array(icons_seg == 2, dtype=int)

    doors = binary_dilation(doors,square(5))
    # doors = binary_closing(doors,square(5))

    floor_plan = walls | doors # '|' = 'OR operator'
    if args.visualize:
        cv2.imwrite(f'{image_folder}pr_mask.png', mask*255)
        cv2.imwrite(f'{image_folder}pr_walls.png', walls*255)
        cv2.imwrite(f'{image_folder}pr_doors.png', doors*255)
        cv2.imwrite(f'{image_folder}pr_floor_plan.png', floor_plan*255)
    return floor_plan, mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', nargs='?', type=str, default='hg_furukawa_original',
                        help='Architecture to use [\'hg_furukawa_original, cc5k\']')
    parser.add_argument('--data-path', nargs='?', type=str, default='data/cubicasa5k/',
                            help='Path to data directory')
    parser.add_argument('--n-classes', nargs='?', type=int, default=44,
                        help='# of classes [\'27, 29, 44\']')
    parser.add_argument('--lmdb-path', nargs='?', type=str, default='cubi_lmdb/',
                        help='Path to lmdb')
    parser.add_argument('--len-divisor', nargs='?', type=int, default=40,
                    help='Number with which to divide the size of the train and val dataset.')
    parser.add_argument('--re-skeletonize', nargs='?', type=bool, default=True, const=True,
                    help='Redo the skeletonization even if npy file exists.')
    parser.add_argument('--visualize', nargs='?', type=bool, default=True, const=True,
                    help='Save .png images that visualizes the process.')
    # model_best_val_loss_var.pkl
    # runs_cubi/2024-06-13-09:11:56-nclass:27-freeze-cc5k/model_best_val_loss_var.pkl
    parser.add_argument('--weights', nargs='?', type=str, default='model_best_val_loss_var.pkl',
                        help='Path to previously trained model weights file .pkl')
    args = parser.parse_args()

    find_skeleton(args)
