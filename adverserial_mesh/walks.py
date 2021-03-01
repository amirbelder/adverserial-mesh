import os

from easydict import EasyDict
import numpy as np

import utils
from sklearn.neighbors import BallTree


def jump_to_closest_unviseted(model_kdtree_query, model_n_vertices, walk, enable_super_jump=True):
  for nbr in model_kdtree_query[walk[-1]]:
    if nbr not in walk:
      return nbr

  if not enable_super_jump:
    return None

  # If not fouind, jump to random node
  node = np.random.randint(model_n_vertices)

  return node


def get_seq_random_walk_no_jumps(mesh_extra, f0, seq_len):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  backward_steps = 1
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    if len(nodes_to_consider):
      to_add = np.random.choice(nodes_to_consider)
      jump = False
    else:
      if i > backward_steps:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
      else:
        to_add = np.random.randint(n_vertices)
        jump = True
    assert to_add > -1
    seq[i] = to_add
    jumps[i] = jump
    visited[to_add] = 1

  return seq, jumps


def get_seq_random_walk_random_global_jumps(mesh_extra, f0, seq_len):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  backward_steps = 1
  jump_prob = 1 / 100
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    jump_now = np.random.binomial(1, jump_prob)
    # jump_now = (i+1) % jump_every_k
    if len(nodes_to_consider) and not jump_now:
      to_add = nodes_to_consider[np.random.randint(len(nodes_to_consider))]
      jump = False
      backward_steps = 1
    else:
      if i > backward_steps and not jump_now:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
      else:
        to_add = np.random.randint(n_vertices)
        jump = True
        visited[...] = 0
        visited[-1] = True
        backward_steps = 1
        last_jump = i
    visited[to_add] = 1
    seq[i] = to_add
    jumps[i] = jump

  return seq, jumps


def get_seq_random_walk_euclidean_jumps(mesh_extra, f0, seq_len):
  '''
  Return random walk with jump every K steps, jumping with probability according to distance from current position!
  :param mesh_extra:
  :param f0:
  :param seq_len:
  :return:
  '''
  vertices = mesh_extra['vertices']
  ball_tree = BallTree(vertices)

  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  backward_steps = 1
  jump_indices = 50
  last_jump = 0
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    jump_now = (i+1) % jump_indices == 0
    if len(nodes_to_consider) and not jump_now:
      to_add = np.random.choice(nodes_to_consider)
      jump = False
      backward_steps = 1
    else:
      if i > backward_steps - last_jump and not jump_now:
        # TODO: try not to jump if not jump now
        # if i > backward_steps and not jump_now:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
        # print(backward_steps)
      else:
        inds = ball_tree.query_radius(vertices[seq[i-1]].reshape(1, -1), r=0.5)[0]  # indices of neighbors with radius 0.5
        to_add = inds[np.random.randint(len(inds))]  # choose randomly from neighbors in radius
        jump = True
        visited[...] = 0
        visited[-1] = True
        backward_steps = 1
        last_jump = i
    visited[to_add] = 1
    seq[i] = to_add
    jumps[i] = jump

  return seq, jumps

  return


def get_seq_random_walk_random_global_jumps_new(mesh_extra, f0, seq_len):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited_perm =  np.zeros((n_vertices + 1,), dtype=np.bool)
  visited_perm[-1] = True
  visited_perm[f0] = True
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  backprop_stack = []
  jump_prob = 1 / 100
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    # jump_now = np.random.binomial(1, jump_prob)
    jump_now = (i + 1) % 20 == 0
    if len(nodes_to_consider) and not jump_now:
      to_add = np.random.choice(nodes_to_consider)
      jump = False
      backprop_stack.append(i-1)
      # backward_steps = 1
    else:
      if len(backprop_stack) and not jump_now:
        to_add = seq[backprop_stack.pop()]
      else:
        backprop_stack = []
        visited_perm |= visited
        available_indices = np.where(visited_perm==False)[0]
        if len(available_indices):
          to_add = available_indices[np.random.randint(len(available_indices))]
        else:
          to_add = np.random.randint(n_vertices)
          visited_perm = np.zeros_like(visited)
          visited_perm[-1] = True
        jump = True
        visited[...] = 0
        visited[-1] = True
    visited[to_add] = 1
    seq[i] = to_add
    jumps[i] = jump

  return seq, jumps


def get_seq_random_walk_local_jumps(mesh_extra, f0, seq_len):
  n_vertices = mesh_extra['n_vertices']
  kdtr = mesh_extra['kdtree_query']
  seq = np.zeros((seq_len + 1, ), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  seq[0] = f0
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  for i in range(1, seq_len + 1):
    b = min(0, i - 20)
    to_consider = [n for n in kdtr[seq[i - 1]] if not visited[n]]
    if len(to_consider):
      seq[i] = np.random.choice(to_consider)
      jumps[i] = False
    else:
      seq[i] = np.random.randint(n_vertices)
      jumps[i] = True
      visited = np.zeros((n_vertices + 1,), dtype=np.bool)
      visited[-1] = True
    visited[seq[i]] = True

  return seq, jumps


def get_seq_random_walk_constant_global_jumps(mesh_extra, f0, seq_len):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  backprop_inds = []
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  jump_every_k = get_seq_random_walk_constant_global_jumps.k
  cur_len = 0
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    jump_now = (i+1) % jump_every_k == 0
    if len(nodes_to_consider) and not jump_now:
      to_add = np.random.choice(nodes_to_consider)
      jump = False
      backprop_inds.append(i-1)
    else:
      if backprop_inds and not jump_now:
        to_add = seq[backprop_inds.pop()]
      else:
        to_add = np.random.randint(n_vertices)
        jump = True
        visited[...] = 0
        visited[-1] = True
        if jump_now:
          cur_len = 0
          backprop_inds = []
    cur_len += 1
    visited[to_add] = 1
    seq[i] = to_add
    jumps[i] = jump

  return seq, jumps



def get_mesh():
  from dataset_prepare import prepare_edges_and_kdtree, load_mesh, remesh
  from dataset import norm_model
  model_fn = os.path.expanduser('~') + '/Databases/ModelNet40/airplane/test/*690*.off'
  import glob
  model_fn = np.random.choice(glob.glob(model_fn))
  mesh = load_mesh(model_fn)
  mesh, _, _ = remesh(mesh, 4000)
  mesh = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'n_faces_orig': np.asarray(mesh.triangles).shape[0]})
  prepare_edges_and_kdtree(mesh)
  vertices = mesh['vertices']
  norm_model(vertices)
  mesh['vertices'] = vertices
  mesh['n_vertices'] = mesh['vertices'].shape[0]
  return mesh

def show_walk_on_mesh(mesh, i):
  # walk, jumps = get_seq_random_walk_no_jumps(mesh, f0=0, seq_len=400)
  walk, jumps = get_seq_random_walk_random_global_jumps(mesh, f0=0, seq_len=200)

  # get_seq_random_walk_constant_global_jumps.k = 10
  # walk, jumps = get_seq_random_walk_constant_global_jumps(mesh, f0=0, seq_len=400)
  # walk = [walk[i:i + 10] for i in range(1, len(walk), 10)]
  # walk = walk[:9]  # not enough colors in color map, and enough to visualize

  vertices = mesh['vertices']
  if 0:
    dxdydz = np.diff(vertices[walk], axis=0)
    for i, title in enumerate(['dx', 'dy', 'dz']):
      plt.subplot(3, 1, i + 1)
      plt.plot(dxdydz[:, i])
      plt.ylabel(title)
    plt.suptitle('Walk features on Human Body')
  save_name = '/home/ran/mesh_walker/examples'
  utils.visualize_model(mesh['vertices'], mesh['faces'],
                               line_width=1.2, show_edges=1, edge_color_a='black',
                               show_vertices=False, opacity=0.8,
                               point_size=4, all_colors='cadetblue',
                               walk=walk, edge_colors='magenta',
                               off_screen=True, save_fn=os.path.join(save_name, 'airplane_690_walk_{}.png'.format(i)))
  print('debug')


if __name__ == '__main__':
  utils.config_gpu(False)
  np.random.seed(4)
  for i in range(16):
    mesh = get_mesh()
    show_walk_on_mesh(mesh, i)
  print('debug')
