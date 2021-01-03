import glob, os
import time

import tensorflow as tf
from tensorflow_graphics.notebooks import mesh_segmentation_dataio as dataio
import numpy as np

import utils_tfg

MAX_NEIGHBORS = 20      # Max number of neighbors per vertex

DEFAULT_IO_PARAMS = {
  'batch_size': 1,
  'shuffle_buffer_size': 100,
  'is_training': True,
  'parallel_threads': 5,
  'mean_center': True,
  'shuffle': None,
  'repeat': None,
}


def get_random_walk(neighbors, n_vertices, walk_length):
  '''
  Generate a random walk (list of vertex indices)
  :param neighbors: np.array. |V| X MAX_NEIGHBORS . If number neighbors < MAX_NEIGHBORS, fill it with -1.
  :param n_vertices: |V|
  :param walk_length: desired number of vertices in walk
  :return: np.array of a random walk
  '''
  f0 = np.random.randint(n_vertices)
  seq = np.zeros((walk_length,), dtype=np.int32)
  visited = np.zeros((n_vertices,), dtype=np.bool)
  visited[-1] = True # last nbrs are set to -1
  visited[f0] = True
  seq[0] = f0
  backward_steps = 1
  jump_prob = 1 / 100
  for i in range(1, walk_length):
    this_nbrs = neighbors[seq[i - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    jump_now = np.random.binomial(1, jump_prob)
    if len(nodes_to_consider) and not jump_now:   # If we have near unvisited nodes
      to_add = np.random.choice(nodes_to_consider)
      backward_steps = 1                          # ?
    else:
      if i > backward_steps and not jump_now:     # in case of a dead-end, start going back
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
      else:
        to_add = np.random.randint(n_vertices)
        visited[...] = 0                          # In case of a jump, clear the visited array
        visited[-1] = True
    visited[to_add] = 1
    seq[i] = to_add

  return seq


def fill_dxdydz_features(features, vertices, walk, walk_length):
  features[1:] = np.diff(vertices[walk[:walk_length]], axis=0) * 100


def get_walks_from_mesh(vertices, neighbors, walk_length, n_walks_per_mesh, n_vertices):
  walk_representation = np.zeros((n_walks_per_mesh, walk_length, 3), dtype=np.float32)
  walks   = np.zeros((n_walks_per_mesh, walk_length), dtype=np.int32)

  for walk_id in range(n_walks_per_mesh):
    walk = get_random_walk(neighbors, n_vertices, walk_length)
    fill_dxdydz_features(walk_representation[walk_id], vertices, walk, walk_length)
    walks[walk_id] = walk

  return walk_representation, walks


def edges2neighbors(edges, n_vertices, MAX_NEIGHBORS):
  def _add_nbr(e0, e1):
    if e0 != e1 and n_nbrs[e0] < MAX_NEIGHBORS and e1 not in neighbors[e0]:
      neighbors[e0, n_nbrs[e0]] = e1
      n_nbrs[e0] += 1

  neighbors = -np.ones((n_vertices, MAX_NEIGHBORS), dtype=np.int32)
  n_nbrs = np.zeros((n_vertices,), dtype=np.int32)
  for edge in edges:
    _add_nbr(edge[0], edge[1])
    _add_nbr(edge[1], edge[0])

  return neighbors


glb_neighbors = {} # Cache neighbors calculation to speed-up runtime
def example_to_walks_representation(example, walk_length=10, n_walks_per_mesh=4):
  n_shapes = example['vertices'].shape[0]
  assert n_shapes == 1, 'For each batch, currently only one shape can be used'
  n_vertices = example['num_vertices'][0]
  vertices = example['vertices'][0].numpy()
  h = hash(vertices.tostring())
  if 'neighbors' in example:
    neighbors = example['neighbors'][0].numpy()
  elif h in glb_neighbors.keys():
    neighbors = glb_neighbors[h]
  else:
    edges = example['edges'][0]
    neighbors = edges2neighbors(edges.numpy(), n_vertices, MAX_NEIGHBORS)
    glb_neighbors[h] = neighbors
  walks, walks_vertices = get_walks_from_mesh(vertices, neighbors, walk_length, n_walks_per_mesh, n_vertices)

  return walks, walks_vertices


def get_train_params():
  train_io_params = {
    'batch_size': 1,
    'parallel_threads': 8,
    'is_training': True,
    'shuffle': True,
    'sloppy': True,
  }

  path_to_train_data_zip = tf.keras.utils.get_file(
    'train_data.zip',
    origin='https://storage.googleapis.com/tensorflow-graphics/notebooks/mesh_segmentation/train_data.zip',
    extract=True)
  train_data_files = glob.glob(
    os.path.join(os.path.dirname(path_to_train_data_zip), '*train*.tfrecords'))

  return train_io_params, train_data_files


def get_test_params():
  path_to_data_zip = tf.keras.utils.get_file(
    'data.zip',
    origin='https://storage.googleapis.com/tensorflow-graphics/notebooks/mesh_segmentation/data.zip',
    extract=True)
  test_data_files = [
    os.path.join(
      os.path.dirname(path_to_data_zip),
      'data/Dancer_test_sequence.tfrecords')
  ]
  eval_io_params = {
    'batch_size': 1,
    'parallel_threads': 8,
    'is_training': False,
    'shuffle': False
  }

  return eval_io_params, test_data_files


def get_dancer_dataset(is_training):
  if is_training:
    params, tfrecord_files = get_train_params()
  else:
    params, tfrecord_files = get_test_params()
  for k in DEFAULT_IO_PARAMS:
    params[k] = params[k] if k in params else DEFAULT_IO_PARAMS[k]
  dataset = dataio.create_dataset_from_tfrecords(tfrecord_files, params)

  return dataset


def get_dataset(dataset_name, is_training):
  if dataset_name == 'dancer':
    dataset = get_dancer_dataset(is_training)
    classification_task = False
    n_classes = 16
  else:
    raise Exception('Dataset ' + dataset_name + ' is not supported.')

  return dataset, classification_task, n_classes


def show_ds():
  """For debug, show meshes and walks"""
  dataset, classification_task, n_classes = get_dataset('dancer', 1)

  walk_length = 800
  n_walks_per_mesh = 1
  tb = time.time()
  for n, example in enumerate(dataset):
    walks_representations, walks_vertices = example_to_walks_representation(example, walk_length, n_walks_per_mesh=n_walks_per_mesh)
    utils_tfg.visualize_model_walk(example['vertices'].numpy()[0], example['triangles'].numpy()[0], walks=walks_vertices)
    if n > 100:
      break
  print('time:', time.time() - tb)


if __name__ == '__main__':
  utils_tfg.config_gpu(False)
  np.random.seed(1)
  show_ds()
