import glob, os, shutil, sys, json
from pathlib import Path

import trimesh
import open3d
from easydict import EasyDict
import numpy as np
from tqdm import tqdm

import pywavefront as pwf
from plyfile import PlyData
from collections import defaultdict
from sklearn.neighbors import BallTree
from typing import List
from shutil import copyfile

# import utils

# Labels for all datasets
# -----------------------

s3dis_labels = ['ceiling', 'floor', 'wall', 'beam', 'column',
                'window', 'door', 'table', 'chair', 'sofa',
                'bookcase', 'board', 'clutter', '<UNK>']
s3dis_shape2label = {v: k for k, v in enumerate(s3dis_labels)}
s3dis_shape2label['<UNK>'] = -1

scannet_v2_labels = ['unannotated', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain',
               'toilet', 'sink', 'bathtub', 'otherfurniture']
scannet_v2_shape2label = {v: k for k, v in enumerate(scannet_v2_labels)}
scannet_v2_colors = [[255, 255, 255],  # unlabeled
                     [174, 199, 232],  # wall
                     [152, 223, 138],  # floor
                     [31, 119, 180],  # cabinet
                     [255, 187, 120],  # bed
                     [188, 189, 34],  # chair
                     [140, 86, 75],  # sofa
                     [255, 152, 150],  # table
                     [214, 39, 40],  # door
                     [197, 176, 213],  # window
                     [148, 103, 189],  # bookshelf
                     [196, 156, 148],  # picture
                     [23, 190, 207],  # counter
                     [247, 182, 210],  # desk
                     [219, 219, 141],  # curtain
                     [255, 127, 14],  # refrigerator
                     [158, 218, 229],  # shower curtain
                     [44, 160, 44],  # toilet
                     [112, 128, 144],  # sink
                     [227, 119, 194],  # bathtub
                     [82, 84, 163]]  # otherfurn
scannet_weights = [3.508061818168880297e+00,
                  4.415242725535003743e+00,
                  1.929816058226905895e+01,
                  2.628740008695115193e+01,
                  1.212917345982307893e+01,
                  2.826658055253028934e+01,
                  2.148932725385034459e+01,
                  1.769486222014486643e+01,
                  1.991481374929695747e+01,
                  2.892054111644061365e+01,
                  6.634054658350238753e+01,
                  6.669804496207542854e+01,
                  3.332619576690268559e+01,
                  3.076747790368030167e+01,
                  6.492922584696864874e+01,
                  7.542849603844955197e+01,
                  7.551157920875556329e+01,
                  7.895305324715594963e+01,
                  7.385072181024294480e+01,
                  2.166310943989462956e+01]


SCANNET_CLASS_REMAP = np.zeros(41)

SCANNET_CLASS_REMAP[1] = 1
SCANNET_CLASS_REMAP[2] = 2
SCANNET_CLASS_REMAP[3] = 3
SCANNET_CLASS_REMAP[4] = 4
SCANNET_CLASS_REMAP[5] = 5
SCANNET_CLASS_REMAP[6] = 6
SCANNET_CLASS_REMAP[7] = 7
SCANNET_CLASS_REMAP[8] = 8
SCANNET_CLASS_REMAP[9] = 9
SCANNET_CLASS_REMAP[10] = 10
SCANNET_CLASS_REMAP[11] = 11
SCANNET_CLASS_REMAP[12] = 12
SCANNET_CLASS_REMAP[14] = 13
SCANNET_CLASS_REMAP[16] = 14
SCANNET_CLASS_REMAP[24] = 15
SCANNET_CLASS_REMAP[28] = 16
SCANNET_CLASS_REMAP[33] = 17
SCANNET_CLASS_REMAP[34] = 18
SCANNET_CLASS_REMAP[36] = 19
SCANNET_CLASS_REMAP[39] = 20


def get_color_and_labels(original_vertices: np.ndarray, representative_vertices: np.ndarray) -> List[np.ndarray]:
  """find nearest neighbor in Euclidean space to interpolate color and label information to vertices in simplified mesh.
  Arguments:
      original_vertices {np.ndarray} -- vertex positions in original mesh
      representative_vertices {np.ndarray} -- vertex positions in simplified mesh
  Returns:
      List[np.ndarray] -- list of arrays containing RGB color and label information
  """
  ball_tree = BallTree(original_vertices[:, :3])
  _, ind = ball_tree.query(representative_vertices, k=1)

  return original_vertices[ind].squeeze()[:, 3:]


def prepare_edges_and_kdtree(mesh):
  vertices = mesh['vertices']
  faces = mesh['faces']
  mesh['edges'] = [set() for _ in range(vertices.shape[0])]
  for i in range(faces.shape[0]):
    for v in faces[i]:
      mesh['edges'][v] |= set(faces[i])
  for i in range(vertices.shape[0]):
    if i in mesh['edges'][i]:
      mesh['edges'][i].remove(i)
    mesh['edges'][i] = list(mesh['edges'][i])
  max_vertex_degree = np.max([len(e) for e in mesh['edges']])
  for i in range(vertices.shape[0]):
    if len(mesh['edges'][i]) < max_vertex_degree:
      mesh['edges'][i] += [-1] * (max_vertex_degree - len(mesh['edges'][i]))
  mesh['edges'] = np.array(mesh['edges'], dtype=np.int32)

  mesh['kdtree_query'] = []
  t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
  n_nbrs = min(10, vertices.shape[0] - 2)
  if n_nbrs < 2:
    n_nbrs=2
  for n in range(vertices.shape[0]):
    d, i_nbrs = t_mesh.kdtree.query(vertices[n], n_nbrs)
    try:
      i_nbrs_cleared = [inbr for inbr in i_nbrs if inbr != n and inbr < vertices.shape[0]]
    except:
      print('catched a problem - debug')
    if len(i_nbrs_cleared) > n_nbrs - 1:
      i_nbrs_cleared = i_nbrs_cleared[:n_nbrs - 1]
    mesh['kdtree_query'].append(np.array(i_nbrs_cleared, dtype=np.int32))
  mesh['kdtree_query'] = np.array(mesh['kdtree_query'])
  assert mesh['kdtree_query'].shape[1] == (n_nbrs - 1), 'Number of kdtree_query is wrong: ' + str(mesh['kdtree_query'].shape[1])

# ------------------------------------------------------- #

def prepare_mesh_scannet(original_vertices, processed_mesh, out_fn):
  if np.asarray(processed_mesh.triangles).shape[0] < 1000:
    return
  processed_mesh.compute_vertex_normals()
  if np.asarray(processed_mesh.vertices).shape[0] < 10:
    print('debug - no points in processed mesh')
  colors_labels = get_color_and_labels(original_vertices, np.asarray(processed_mesh.vertices))
  mesh_data = EasyDict(
    {'vertices': np.asarray(processed_mesh.vertices),
     'faces': np.asarray(processed_mesh.triangles),
     'label': colors_labels[:, -1] - 1,
     'labels': colors_labels[:, -1] - 1,
     'v_rgb': colors_labels[:, :3],
     'vertex_normals': colors_labels[:, 3:6]})
  prepare_edges_and_kdtree(mesh_data)
  np.savez(out_fn, **mesh_data)
  print('saved model {}'.format(out_fn))
  print('n_vertices: {}\t n_faces: {}'.format(mesh_data['vertices'].shape[0], mesh_data['faces'].shape[0]))


def handle_single_crop(original_vertices, simplified_mesh, cur_pos, box_size, min_z, max_z, out_fn):
  min_pos = (cur_pos[0] - box_size / 2, cur_pos[1] - box_size / 2, min_z)
  max_pos = (cur_pos[0] + box_size / 2, cur_pos[1] + box_size / 2, max_z)
  cropped_mesh = simplified_mesh.crop(min_pos, max_pos)
  # add required fields and save cropped mesh

  prepare_mesh_scannet(original_vertices, cropped_mesh, out_fn)


def prepare_scannet_v2():
  p_out = os.path.expanduser('~') + '/mesh_walker/datasets/scannet_v2_4cm_crops/'
  if not os.path.exists(p_out):
    os.makedirs(p_out)
  for part in ['test']:   #['train', 'val', 'test']:
    considered_rooms_path = 'meta/scannetv2_{}.txt'.format(part)
    with open(considered_rooms_path, 'r') as f:
      considered_rooms = f.read().splitlines()
      if part == 'test':
        fps = glob.glob(f"{'/home/ran/Databases/scannet_v2/scans_test'}/*/*.ply")
      else:
        fps = glob.glob(f"{'/home/ran/Databases/scannet_v2/scans'}/*/*.ply")
      file_paths = sorted([x for
                           x in fps
                           if 'clean_2.ply' in x
                           and x.split('/')[-1].rsplit('_', 3)[0] in considered_rooms])
      for file_path in file_paths:
        out_fn = p_out + part + '_' + file_path.split('/')[-2]
        original_mesh = open3d.io.read_triangle_mesh(file_path)  #TODO: verify it keeps order -
        original_mesh.compute_vertex_normals()
        if part in ['train', 'val']:
          labels_file_path = file_path.replace('.ply', '.labels.ply')
          vertex_labels = np.asarray(PlyData.read(
            labels_file_path)['vertex']['label'])
          # zero_map = np.zeros((vertex_labels.shape[0], 3), np.int)
          # zero_map[:, 0] += vertex_labels
          # labels_mesh.vertex_colors = open3d.utility.Vector3dVector(zero_map)
          # simplified_label_mesh = labels_mesh.simplify_vertex_clustering(voxel_size=0.04,
          #                                                            contraction=open3d.geometry.SimplificationContraction.Average)
          # simplified_vertex_labels = np.asarray(simplified_label_mesh.vertex_colors)[:,0]

          original_vertices = np.column_stack(
            (np.asarray(original_mesh.vertices),
             np.asarray(original_mesh.vertex_colors),
             np.asarray(original_mesh.vertex_normals),
             vertex_labels))


          # FIX: THREE MESHES HAVE CORRUPTED LABEL IDS
          class_ids = original_vertices[:, -1].astype(int)
          class_ids[class_ids > 40] = 0
          original_vertices[:, -1] = SCANNET_CLASS_REMAP[class_ids]

        else:
          # Test set - no labels
          original_vertices = np.column_stack((np.asarray(original_mesh.vertices),
                                               np.asarray(
                                                 original_mesh.vertex_colors),
                                               np.asarray(original_mesh.vertex_normals),
                                               np.zeros(np.asarray(original_mesh.vertices).shape[0])))

        # perform vertex clustering with 0.04cm
        simplified_mesh = original_mesh.simplify_vertex_clustering(voxel_size=0.04,
                                                                   contraction=open3d.geometry.SimplificationContraction.Average)


        if part == 'train':
          # Perform mesh crop for training examples - with overlaps - ON TRAIN SET ALONE
          # get crops every area, of 3x3xZ (full size in height). same as https://arxiv.org/pdf/2004.01002.pdf
          stride = 1.5  # meters
          box_size = 3.0
          start_pos = np.min(np.asarray(simplified_mesh.vertices), axis=0) + stride
          last_pos = np.max(np.asarray(simplified_mesh.vertices), axis=0) - stride
          min_z =  start_pos[-1] - stride
          max_z =  last_pos[-1] + stride
          crop_idx=0

          cur_pos = start_pos
          while cur_pos[0] < last_pos[0]:
            cur_pos[1] = start_pos[1]
            while cur_pos[1] < last_pos[1]:
              # crop with open3d crop function , remember to manually add last_pos! entire row + column! (since it is not exact division of stride)
              cropped_fn = out_fn + '_crop_{:04d}'.format(crop_idx)
              handle_single_crop(original_vertices, simplified_mesh, cur_pos, box_size, min_z, max_z, cropped_fn)

              # === Advance Y pos === #
              crop_idx += 1
              cur_pos[1] += stride

            # handle last Y step manually to catch edge of scene
            cur_pos[1] = last_pos[1]
            cropped_fn = out_fn + '_crop_{:04d}'.format(crop_idx)
            handle_single_crop(original_vertices, simplified_mesh, cur_pos, box_size, min_z, max_z,
                               cropped_fn)

            # Advance X pos
            crop_idx += 1
            cur_pos[0] += stride

          # handle last X pos (with all Ys pos with it) manuallt
          cur_pos[0] = last_pos[0]
          while cur_pos[1] < last_pos[1]:
            # crop with open3d crop function , remember to manually add last_pos! entire row + column! (since it is not exact division of stride)
            cropped_fn = out_fn + '_crop_{:04d}'.format(crop_idx)
            handle_single_crop(original_vertices, simplified_mesh, cur_pos, box_size, min_z, max_z,
                               cropped_fn)
            crop_idx += 1
            cur_pos[1] += stride

          # handle last Y step manually to catch edge of scene
          cur_pos[1] = last_pos[1]
          cropped_fn = out_fn + '_crop_{:04d}'.format(crop_idx)
          handle_single_crop(original_vertices, simplified_mesh, cur_pos, box_size, min_z, max_z,
                           cropped_fn)


        else:
          prepare_mesh_scannet(original_vertices, simplified_mesh, out_fn)

# ===------------ #

# ------------------------------------------------------- #


def visualize_segmentation_dataset(pathname_expansion):
  cpos = None
  filenames = glob.glob(pathname_expansion)
  while 1:
    fn = np.random.choice(filenames)
    mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
    import utils
    cpos = utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=mesh_data['labels'].astype(np.int), cpos=cpos)


if __name__ == '__main__':
  TEST_FAST = 0
  # utils.config_gpu(False)
  np.random.seed(1)

  prepare_scannet_v2()
