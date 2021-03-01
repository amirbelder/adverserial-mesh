import glob, os, copy

import tensorflow as tf
import numpy as np

import utils
import walks
import dataset_prepare


# Glabal list of dataset parameters
dataset_params_list = []

def get_far_points(vertices, n_points=1):
  f0_list = []
  # vert_mean = np.mean(vertices,axis=0)
  # vert_dists = np.linalg.norm(vertices - np.repeat(np.expand_dims(vert_mean, axis=0), vertices.shape[0], 0), axis=1)
  # max_verts = np.argmax(vert_dists)
  f0_list.append(np.random.randint(vertices.shape[0]))
  while len(f0_list) < n_points:
    # add points by max distance from current chosen points
    # Choose K random points
    K= vertices.shape[0]
    k_verts = np.random.permutation(K)
    k_dists = np.asarray([np.linalg.norm(vertices[k_verts] - np.repeat(np.expand_dims(x, axis=0), K, 0), axis=1) for x in vertices[f0_list]])
    if len(k_dists.shape) > 1:
      # k_dists = np.sum(k_dists, axis=0)
      k_dists = np.min(k_dists, axis=0)
    f0_list.append(k_verts[np.argmax(k_dists)])
  return f0_list


def load_model_from_npz(npz_fn):
  if npz_fn.find(':') != -1:
    npz_fn = npz_fn.split(':')[1]
  mesh_data = np.load(npz_fn, encoding='latin1', allow_pickle=True)
  return mesh_data


def norm_model(vertices, return_val=False):
  # Move the model so the bbox center will be at (0, 0, 0)
  mean = np.mean((np.min(vertices, axis=0), np.max(vertices, axis=0)), axis=0)
  vertices -= mean

  # Scale model to fit into the unit ball
  if 1:
    norm_with = np.max(vertices)
  else:
    norm_with = np.max(np.linalg.norm(vertices, axis=1))
  vertices /= norm_with

  if hasattr(norm_model, 'sub_mean_for_data_augmentation') and norm_model.sub_mean_for_data_augmentation:
    vertices -= np.nanmean(vertices, axis=0)
  if return_val:
    return vertices

def norm_model_scannet(vertices):
  vertices[:, 0] -= np.mean((np.min(vertices[:, 2]), np.max(vertices[:, 2])))
  vertices[:, 1] -= np.mean((np.min(vertices[:, 0]), np.max(vertices[:, 0])))
  vertices[:, 2] -= np.min(vertices[:, 1])


def data_augmentation_axes_rot(vertices):
  if np.random.randint(2):    # 50% chance to switch the two hirisontal axes
    vertices[:] = vertices[:, data_augmentation_axes_rot.flip_axes]
  if np.random.randint(2):    # 50% chance to neg one random hirisontal axis
    i = np.random.choice(data_augmentation_axes_rot.hori_axes)
    vertices[:, i] = -vertices[:, i]


def rotate_to_check_weak_points(max_rot_ang_deg):
  if np.random.randint(2):
    x = max_rot_ang_deg
  else:
    x = -max_rot_ang_deg
  if np.random.randint(2):
    y = max_rot_ang_deg
  else:
    y = -max_rot_ang_deg
  if np.random.randint(2):
    z = max_rot_ang_deg
  else:
    z = -max_rot_ang_deg

  return x, y, z


def data_augmentation_rotation(vertices):
  if 1:#np.random.randint(2):    # 50% chance
    max_rot_ang_deg = data_augmentation_rotation.max_rot_ang_deg
    if 0:
      x = y = z = 0
      if data_augmentation_rotation.test_rotation_axis == 0:
        x = max_rot_ang_deg
      if data_augmentation_rotation.test_rotation_axis == 1:
        y = max_rot_ang_deg
      if data_augmentation_rotation.test_rotation_axis == 2:
        z = max_rot_ang_deg
    else:
      x = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
      y = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
      z = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
    A = np.array(((np.cos(x), -np.sin(x), 0),
                  (np.sin(x), np.cos(x), 0),
                  (0, 0, 1)),
                 dtype=vertices.dtype)
    B = np.array(((np.cos(y), 0, -np.sin(y)),
                  (0, 1, 0),
                  (np.sin(y), 0, np.cos(y))),
                 dtype=vertices.dtype)
    C = np.array(((1, 0, 0),
                  (0, np.cos(z), -np.sin(z)),
                  (0, np.sin(z), np.cos(z))),
                 dtype=vertices.dtype)
    np.dot(vertices, A, out=vertices)
    np.dot(vertices, B, out=vertices)
    np.dot(vertices, C, out=vertices)


def data_augmentation_const_rotation(vertices):
    angle = np.random.choice(np.linspace(-180, 180, int(360 / 30), endpoint=False))
    y = angle * np.pi / 180

    R = np.array(((np.cos(y),-np.sin(y), 0),
                  (np.sin(y), np.cos(y), 0),
                  (0        ,         0, 1)),
                 dtype=vertices.dtype)
    np.dot(vertices, R, out=vertices)


def data_augmentation_aspect_ratio(vertices):
  if np.random.randint(2):    # 50% chance
    for i in range(3):
      r = np.random.uniform(1 - data_augmentation_aspect_ratio.max_ratio, 1 + data_augmentation_aspect_ratio.max_ratio)
      vertices[i] *= r


def data_augmentation_translation(vertices):
  if np.random.randint(2):
    for i in range(3):
      t = np.random.uniform(-data_augmentation_translation.max,data_augmentation_translation.max)
      vertices[i] +=t


def data_augmentation_jitter(vertices):
  clip = 0.1
  noise = np.clip(np.random.normal(0., data_augmentation_jitter.sigma, vertices.shape), -1 * clip, clip)
  vertices += noise
  return vertices


def fill_xyz_features(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  walk = vertices[seq[1:seq_len + 1]]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 3
  return f_idx


def fill_rgb_features(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  walk = mesh_extra['v_rgb'][seq[1:seq_len + 1]]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 3
  return f_idx


def fill_vertex_normals_features(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  '''
  # TODO: if we augment with rotations, vertex normals should be rotated too
  #      (we rotated points, these are pre-calculated)
  :param features:
  :param f_idx:
  :param vertices:
  :param mesh_extra:
  :param seq:
  :param jumps:
  :param seq_len:
  :return:
  '''
  walk = mesh_extra['vertex_normals'][seq[1:seq_len + 1]]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 3
  return f_idx


def fill_dxdydz_features(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  walk = np.diff(vertices[seq[:seq_len + 1]], axis=0) * 100
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 3
  return f_idx


def fill_fpf_features(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  '''
  returns alpha, phi, theta as descriped in Fast Point Feature histogram (Rusu et. al. 2009)
  Difference is we compute only between previous point to current point in walk - hence we get information about the
  change of normals orientation along the curve we create.
  Normals are pre-computed from FACES - so normal of vertex do contain small are (neighbor faces) information
  :param features:
  :param f_idx:
  :param vertices:
  :param mesh_extra:
  :param seq:
  :param jumps:
  :param seq_len:
  :return: fills features with seq_len X (alpha, phi, theta)
  '''
  #
  walk_normals =  mesh_extra['vertex_normals'][seq[:seq_len+1]]
  walk_dxdydz = np.diff(vertices[seq[:seq_len + 1]], axis=0)
  normed_dxdydz = walk_dxdydz / np.expand_dims(np.linalg.norm(walk_dxdydz, axis=-1) + 1e-6, axis=-1)
  u = walk_normals[:seq_len]
  v = np.cross(normed_dxdydz, u)
  w = np.cross(u, v)
  nj = walk_normals[1:]
  alphas = np.sum(v * nj, axis=-1)
  phi = np.sum(u * normed_dxdydz, axis=-1)
  theta = np.arctan2(np.sum(w*nj, axis=-1), np.sum(u*nj, axis=-1))
  features[:, f_idx:f_idx + 3] = np.stack([alphas, phi, theta], axis=-1)
  f_idx += 3
  return f_idx


def fill_fpfh_features(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  '''
  returns Pre-computed fpfh as descriped in Fast Point Feature histogram (Rusu et. al. 2009)
  must have key 'fpfh' in mesh_extra
  :param features:
  :param f_idx:
  :param vertices:
  :param mesh_extra:
  :param seq:
  :param jumps:
  :param seq_len:
  :return: fills features with seq_len X (alpha, phi, theta)
  '''
  #
  walk = mesh_extra['mfpfh'][seq[1:seq_len + 1]]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 33
  return f_idx


def fill_vertex_indices(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  walk = seq[1:seq_len + 1][:, None]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 1
  return f_idx


def fill_jumps(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  walk = jumps[1:seq_len + 1][:, None]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 1
  return f_idx


def setup_data_augmentation(dataset_params, data_augmentation):
  dataset_params.data_augmentaion_vertices_functions = []
  if 'horisontal_90deg' in data_augmentation.keys() and data_augmentation['horisontal_90deg']:
    dataset_params.data_augmentaion_vertices_functions.append(data_augmentation_axes_rot)
    data_augmentation_axes_rot.hori_axes = data_augmentation['horisontal_90deg']
    flip_axes_ = [0, 1, 2]
    data_augmentation_axes_rot.flip_axes  = [0, 1, 2]
    data_augmentation_axes_rot.flip_axes[data_augmentation_axes_rot.hori_axes[0]] = flip_axes_[data_augmentation_axes_rot.hori_axes[1]]
    data_augmentation_axes_rot.flip_axes[data_augmentation_axes_rot.hori_axes[1]] = flip_axes_[data_augmentation_axes_rot.hori_axes[0]]
  if 'rotation' in data_augmentation.keys() and data_augmentation['rotation']:
    data_augmentation_rotation.max_rot_ang_deg = data_augmentation['rotation']
    dataset_params.data_augmentaion_vertices_functions.append(data_augmentation_rotation)
  if 'const_rotation' in data_augmentation.keys() and data_augmentation['const_rotation']:
    data_augmentation_const_rotation.rot_ang_deg = data_augmentation['const_rotation']
    dataset_params.data_augmentaion_vertices_functions.append(data_augmentation_const_rotation)

  if 'aspect_ratio' in data_augmentation.keys() and data_augmentation['aspect_ratio']:
    data_augmentation_aspect_ratio.max_ratio = data_augmentation['aspect_ratio']
    dataset_params.data_augmentaion_vertices_functions.append(data_augmentation_aspect_ratio)
  if 'translation' in data_augmentation.keys() and data_augmentation['translation']:
    data_augmentation_translation.max = data_augmentation['translation']
    dataset_params.data_augmentaion_vertices_functions.append(data_augmentation_translation)
  if 'jitter' in data_augmentation.keys() and data_augmentation['jitter']:
    data_augmentation_jitter.sigma = data_augmentation['jitter']
    dataset_params.data_augmentaion_vertices_functions.append(data_augmentation_jitter)


def setup_features_params(dataset_params, params):
  if params.uniform_starting_point:
    dataset_params.area = 'all'
  else:
    dataset_params.area = -1
  norm_model.sub_mean_for_data_augmentation = params.sub_mean_for_data_augmentation
  dataset_params.support_mesh_cnn_ftrs = False
  dataset_params.f_normals = False
  dataset_params.v_normals = False
  dataset_params.use_faces = False
  dataset_params.use_fpfh = False
  dataset_params.use_rgb = False
  dataset_params.vertices_needed = False
  dataset_params.fill_features_functions = []
  dataset_params.number_of_features = 0
  net_input = params.net_input
  if 'xyz' in net_input:
    dataset_params.fill_features_functions.append(fill_xyz_features)
    dataset_params.number_of_features += 3
  if 'dxdydz' in net_input:
    dataset_params.fill_features_functions.append(fill_dxdydz_features)
    dataset_params.number_of_features += 3
  # if 'edge_meshcnn' in net_input:
  #   dataset_params.support_mesh_cnn_ftrs = True
  #   dataset_params.fill_features_functions.append(fill_edge_meshcnn_features)
  #   dataset_params.number_of_features += 5
  # if 'normals' in net_input:
  #   dataset_params.fill_features_functions.append(fill_face_normals_features)
  #   dataset_params.number_of_features += 3
  if 'v_normals' in net_input:
    dataset_params.v_normals = True
    dataset_params.fill_features_functions.append(fill_vertex_normals_features)
    dataset_params.number_of_features += 3
  # if 'rgb' in net_input:
  #   dataset_params.use_rgb = True
  #   dataset_params.fill_features_functions.append(fill_rgb_features)
  #   dataset_params.number_of_features += 3
  if 'jump_indication' in net_input:
    dataset_params.fill_features_functions.append(fill_jumps)
    dataset_params.number_of_features += 1
  # if 'fpfh' in net_input:
  #   dataset_params.v_normals = True
  #   dataset_params.use_fpfh = True
  #   dataset_params.fill_features_functions.append(fill_fpfh_features)
  #   dataset_params.number_of_features += 33
  # if 'fpf' in net_input:
  #   dataset_params.v_normals = True
  #   dataset_params.fill_features_functions.append(fill_fpf_features)
  #   dataset_params.number_of_features += 3
  if 'vertex_indices' in net_input:
    dataset_params.fill_features_functions.append(fill_vertex_indices)
    dataset_params.number_of_features += 1


  dataset_params.edges_needed = True
  if params.walk_alg == 'no_jumps':
    dataset_params.walk_function = walks.get_seq_random_walk_no_jumps
    dataset_params.kdtree_query_needed = False
  elif params.walk_alg == 'random_global_jumps':
    dataset_params.walk_function = walks.get_seq_random_walk_random_global_jumps
    dataset_params.kdtree_query_needed = False
  elif params.walk_alg == 'random_global_jumps_new':
    dataset_params.walk_function = walks.get_seq_random_walk_random_global_jumps_new
    dataset_params.kdtree_query_needed = False
  elif params.walk_alg == 'constant_global_jumps':
    dataset_params.walk_function = walks.get_seq_random_walk_constant_global_jumps
    walks.get_seq_random_walk_constant_global_jumps.k = dataset_params.net_params.jump_every_k
    dataset_params.kdtree_query_needed = False
  elif params.walk_alg == 'local_jumps':
    dataset_params.walk_function = walks.get_seq_random_walk_local_jumps
    dataset_params.kdtree_query_needed = True
    dataset_params.edges_needed = False
  elif params.walk_alg == 'euclidean_jumps':
    dataset_params.walk_function = walks.get_seq_random_walk_euclidean_jumps
    dataset_params.kdtree_query_needed = False
    dataset_params.vertices_needed = True
  else:
    raise Exception('Walk alg not recognized: ' + params.walk_alg)

  return dataset_params.number_of_features


def get_starting_point(area, area_vertices_list, n_vertices, walk_id):
  if area is None or area_vertices_list is None:
    return np.random.randint(n_vertices)
  elif area == -1:
    candidates = np.zeros((0,))
    while candidates.size == 0:
      b = np.random.randint(9)
      candidates = area_vertices_list[b]
    return np.random.choice(candidates)
  else:
    candidates = area_vertices_list[walk_id % len(area_vertices_list)]
    while candidates.size == 0:
      b = np.random.randint(9)
      candidates = area_vertices_list[b]
    return np.random.choice(candidates)


def generate_walk_py_fun(fn, vertices, faces, edges, kdtree_query, v_normals, rgb, tri_centers, mfpfh, labels, params_idx):
  return tf.py_function(
    generate_walk,
    inp=(fn, vertices, faces, edges, kdtree_query, v_normals, rgb, tri_centers, mfpfh, labels, params_idx),
    Tout=(fn.dtype, vertices.dtype, tf.int32)
  )


def generate_ms_walk_py_func(fn, vertices, v_ids, faces, f_ids, edges, e_ids, kdtree_query, v_normals, rgb, tri_centers, mfpfh, labels, params_idx):
  return tf.py_function(
    generate_ms_walk,
    inp=(fn, vertices, v_ids, faces, f_ids, edges, e_ids, kdtree_query, v_normals, rgb, tri_centers, mfpfh, labels, params_idx),
    Tout=(fn.dtype, vertices.dtype, tf.int32)
  )


def generate_ms_walk(fn, vertices, v_ids, faces, f_ids, edges, e_ids, kdtree_query, v_normals, rgb, tri_centers, mfpfh, labels_from_npz, params_idx):
  mesh_data = {'vertices': vertices.numpy(),
               'faces': faces.numpy(),
               'edges': edges.numpy(),
               'kdtree_query': kdtree_query.numpy(),
               'vertex_normals': v_normals.numpy(),
               'v_rgb': rgb.numpy(),
               'tri_centers': tri_centers.numpy(),
               'mfpfh': mfpfh.numpy(),
               'vids': v_ids.numpy(),
               'fids': f_ids.numpy(),
               'eids': e_ids.numpy(),
               }
  if dataset_params_list[params_idx[0]].label_per_step:
    mesh_data['labels'] = labels_from_npz.numpy()

  dataset_params = dataset_params_list[params_idx[0].numpy()]
  features, labels = mesh_data_to_walk_features(mesh_data, dataset_params)

  if dataset_params_list[params_idx[0]].label_per_step:
    labels_return = labels
  else:
    labels_return = labels_from_npz

  return fn[0], features, labels_return


def generate_walk(fn, vertices, faces, edges, kdtree_query, v_normals, rgb, tri_centers, mfpfh, labels_from_npz, params_idx):
  mesh_data = {'vertices': vertices.numpy(),
               'faces': faces.numpy(),
               'edges': edges.numpy(),
               'kdtree_query': kdtree_query.numpy(),
               'vertex_normals': v_normals.numpy(),
               'v_rgb': rgb.numpy(),
               'tri_centers': tri_centers.numpy(),
               'mfpfh': mfpfh.numpy()
               }
  if dataset_params_list[params_idx[0]].label_per_step:
    mesh_data['labels'] = labels_from_npz.numpy()

  dataset_params = dataset_params_list[params_idx[0].numpy()]
  features, labels = mesh_data_to_walk_features(mesh_data, dataset_params)

  if dataset_params_list[params_idx[0]].label_per_step:
    labels_return = labels
  else:
    labels_return = labels_from_npz

  return fn[0], features, labels_return


def mesh_data_to_walk_features(mesh_data, dataset_params):
  vertices = mesh_data['vertices']
  seq_len = dataset_params.seq_len
  if dataset_params.set_seq_len_by_n_faces:
    seq_len = int(mesh_data['vertices'].shape[0])
    seq_len = min(seq_len, dataset_params.seq_len)

  # Preprocessing
  if dataset_params.adjust_vertical_model:
    vertices[:, 1] -= vertices[:, 1].min()
  if dataset_params.normalize_model:
    norm_model(vertices)

  # Data augmentation
  for data_augmentaion_function in dataset_params.data_augmentaion_vertices_functions:
    data_augmentaion_function(vertices)

  # Get essential data from file
  if dataset_params.label_per_step:
    mesh_labels = mesh_data['labels']
  else:
    mesh_labels = -1 * np.ones((vertices.shape[0],))

  mesh_extra = {}
  if hasattr(dataset_params, 'multiscale') and dataset_params.multiscale:
    mesh_extra['n_vertices'] = [mesh_data['vids'][i] - mesh_data['vids'][i-1] for i in range(1, len(mesh_data['vids']))]
    mesh_extra['edges'] = [mesh_data['edges'][mesh_data['vids'][i]:mesh_data['vids'][i+1]] for i in range(len(mesh_data['vids'])-1)]
    vertices = [mesh_data['vertices'][mesh_data['vids'][i]:mesh_data['vids'][i+1]] for i in range(len(mesh_data['vids'])-1)]
    mesh_extra['vertex_normals'] = mesh_data['vertex_normals']
  else:
    mesh_extra['n_vertices'] = vertices.shape[0]
    mesh_extra['vertex_normals'] = mesh_data['vertex_normals']
    # mesh_extra['v_rgb'] = mesh_data['v_rgb']
    # mesh_extra['tri_centers'] = mesh_data['tri_centers']
    # mesh_extra['mfpfh'] = mesh_data['mfpfh']
    if dataset_params.edges_needed:
      mesh_extra['edges'] = mesh_data['edges']
    if dataset_params.kdtree_query_needed:
      mesh_extra['kdtree_query'] = mesh_data['kdtree_query']
    if dataset_params.vertices_needed:
      mesh_extra['vertices'] = mesh_data['vertices']


  features = np.zeros((dataset_params.n_walks_per_model, seq_len, dataset_params.number_of_features), dtype=np.float32)
  labels   = np.zeros((dataset_params.n_walks_per_model, seq_len), dtype=np.int32)

  # # TODO : Scannet re-arrangement, need to make sure we add features and not compromising old ones
  #
  # if hasattr(dataset_params, 'full_coverage') and dataset_params.full_coverage:
  #   features, labels = get_walk_full_coverage(mesh_data, vertices, dataset_params, mesh_extra, seq_len, features, labels)
  # elif dataset_params.network_task == 'self:triplets':
  #   features, labels = get_walk_triplets(mesh_data, vertices, dataset_params, mesh_extra, seq_len, features, labels)
  # else:
  features, labels = get_walk(mesh_data, vertices, dataset_params, mesh_extra, seq_len, features, labels)

  return features, labels



def get_walk_full_coverage(mesh_data, vertices, dataset_params, mesh_extra, seq_len, features, labels):
  '''
  Full coverage might be problematic if many unconnected vertices exists - finding them all with random walks will take
   too much since random walk first try to get on all neighbors - will sample areas before unconnected vertices
  :param mesh_data:
  :param vertices:
  :param dataset_params:
  :param mesh_extra:
  :param seq_len:
  :param features:
  :param labels:
  :return:
  '''
  features = np.zeros((1, seq_len, dataset_params.number_of_features), dtype=np.float32)
  labels = np.zeros((1, seq_len), dtype=np.int32)

  visited_list = np.zeros(vertices.shape[0])
  N_connected = np.count_nonzero(np.unique(mesh_extra['edges']) > -1)
  walk_id = 0
  while np.count_nonzero(visited_list) < N_connected:
    cur_features = np.zeros((1, seq_len, dataset_params.number_of_features), dtype=np.float32)
    unvisited = np.where(visited_list == 0)[0].flatten()
    f0 = unvisited[np.random.randint(len(unvisited))]
    seq, jumps = dataset_params.walk_function(mesh_extra, f0, seq_len)
    visited_list[seq] = 1   #visited_list[seq[len(seq) // 2:]] = 1  # TODO: see if we can check if its rnn or attention - rnn is only latter half of walk
    f_idx = 0
    for fill_ftr_fun in dataset_params.fill_features_functions:
      f_idx = fill_ftr_fun(cur_features[0], f_idx, vertices, mesh_extra, seq, jumps, seq_len)
    if dataset_params.label_per_step:
      mesh_labels = mesh_data['labels']
      cur_labels = np.expand_dims(mesh_labels[seq[1:seq_len + 1]], axis=0)
    if walk_id == 0:
      features = cur_features
      labels = cur_labels
    else:
      features = np.concatenate([features, cur_features], axis=0)
      labels = np.concatenate([labels, cur_labels], axis=0)
    walk_id += 1
  # print('DEBUG - covered with {} walks'.format(walk_id))
  return features, labels


def get_walk(mesh_data, vertices_, dataset_params, mesh_extra_, seq_len, features, labels):
  # if dataset_params.cross_walk_attn:
  #   start_points = get_far_points(vertices, n_points=dataset_params.n_walks_per_model)
  start_points = get_far_points(vertices_, n_points=dataset_params.n_walks_per_model)
  for walk_id in range(dataset_params.n_walks_per_model):
    # if dataset_params.cross_walk_attn:
    #   f0 = start_points[walk_id]
    # else:
    #   f0 = np.random.randint(vertices.shape[0])
    # f0 = start_points[walk_id]
    if hasattr(dataset_params, 'multiscale') and dataset_params.multiscale:
      scale = np.random.randint(len(vertices_))
      vertices = vertices_[scale]
      mesh_extra = copy.deepcopy(mesh_extra_)
      mesh_extra['edges'] = mesh_extra_['edges'][scale]
      mesh_extra['n_vertices'] = mesh_extra_['n_vertices'][scale]
    else:
      vertices = vertices_
      mesh_extra = mesh_extra_
    f0 = np.random.randint(vertices.shape[0])
    # f0 = start_points[walk_id]
    if dataset_params.n_target_vrt_to_norm_walk and dataset_params.n_target_vrt_to_norm_walk < vertices.shape[0]:
      j = int(round(vertices.shape[0] / dataset_params.n_target_vrt_to_norm_walk))
    else:
      j = 1
    seq, jumps = dataset_params.walk_function(mesh_extra, f0, seq_len * j)
    seq = seq[::j]
    if dataset_params.reverse_walk:
      seq = seq[::-1]
      jumps = jumps[::-1]

    f_idx = 0
    for fill_ftr_fun in dataset_params.fill_features_functions:
      f_idx = fill_ftr_fun(features[walk_id], f_idx, vertices, mesh_extra, seq, jumps, seq_len)
    if dataset_params.label_per_step:
      mesh_labels = mesh_data['labels']
      labels[walk_id] = mesh_labels[seq[1:seq_len + 1]]
  return features, labels


def get_walk_triplets(mesh_data, vertices, dataset_params, mesh_extra, seq_len, features, labels):
  neg_walk_f0 = np.random.randint(vertices.shape[0])
  pos_walk_f0 = np.random.choice(mesh_data['far_vertices'][neg_walk_f0])
  for walk_id in range(dataset_params.n_walks_per_model):
    if walk_id < dataset_params.n_walks_per_model / 2:
      f0 = neg_walk_f0
    else:
      f0 = pos_walk_f0
    seq, jumps = dataset_params.walk_function(mesh_extra, f0, seq_len)
    f_idx = 0
    for fill_ftr_fun in dataset_params.fill_features_functions:
      f_idx = fill_ftr_fun(features[walk_id], f_idx, vertices, mesh_extra, seq, jumps, seq_len)
    if dataset_params.label_per_step:
      labels[walk_id] = seq[1:seq_len + 1]
  return features, labels


def get_file_names(pathname_expansion, min_max_faces2use):
  filenames_ = glob.glob(pathname_expansion)
  filenames = []
  for fn in filenames_:
    try:
      n_faces = int(fn.split('.')[-2].split('_')[-1])
      if n_faces > min_max_faces2use[1] or n_faces < min_max_faces2use[0]:
        continue
    except:
      pass
    filenames.append(fn)
  assert len(filenames) > 0, 'DATASET error: no files in directory to be used! \nDataset directory: ' + pathname_expansion

  return filenames


def adjust_fn_list_by_size(filenames_, max_size_per_class):
  lmap = dataset_prepare.map_fns_to_label(filenames=filenames_)
  filenames = []
  if type(max_size_per_class) is int:
    models_already_used = {k: set() for k in lmap.keys()}
    for k, v in lmap.items():
      for i, f in enumerate(v):
        model_name = f.split('/')[-1].split('simplified')[0].split('not_changed')[0]
        if len(models_already_used[k]) < max_size_per_class or model_name in models_already_used[k]:
          filenames.append(f)
          models_already_used[k].add(model_name)
  elif max_size_per_class == 'uniform_as_max_class':
    max_size = 0
    for k, v in lmap.items():
      if len(v) > max_size:
        max_size = len(v)
    for k, v in lmap.items():
      f = int(np.ceil(max_size / len(v)))
      fnms = v * f
      filenames += fnms[:max_size]
  else:
    raise Exception('max_size_per_class not recognized')

  return filenames


def filter_fn_by_class(filenames_, classes_indices_to_use):
  filenames = []
  for fn in filenames_:
    mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
    if classes_indices_to_use is not None and mesh_data['label'] not in classes_indices_to_use:
      continue
    filenames.append(fn)
  return filenames


def setup_dataset_params(params, data_augmentation):
  p_idx = len(dataset_params_list)
  ds_params = copy.deepcopy(params)
  ds_params.set_seq_len_by_n_faces = False
  if 'n_target_vrt_to_norm_walk' not in ds_params.keys():
    ds_params.n_target_vrt_to_norm_walk = 0

  setup_data_augmentation(ds_params, data_augmentation)
  setup_features_params(ds_params, params)

  dataset_params_list.append(ds_params)

  return p_idx


class OpenMeshDataset(tf.data.Dataset):
  # OUTPUT:      (fn,               vertices,          faces,           edges,           kdtree_query,
  OUTPUT_TYPES = (tf.dtypes.string, tf.dtypes.float32, tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.int32,
              #   ( vertices_normals, vertices_rgb,      faces_centers        fpfh               labels          params_idx)
                  tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.int32, tf.dtypes.int32)

  def _generator(fn_, params_idx):
    fn = fn_[0]
    with np.load(fn, encoding='latin1', allow_pickle=True) as mesh_data:
      vertices = mesh_data['vertices']
      faces = mesh_data['faces']
      edges = mesh_data['edges']
      if dataset_params_list[params_idx].v_normals:
        v_normals = mesh_data['vertex_normals']
      else:
        v_normals = [-1]
      if dataset_params_list[params_idx].use_rgb:
        v_rgb = mesh_data['v_rgb']
      else:
        v_rgb = [-1]
      if dataset_params_list[params_idx].use_faces:
        tri_centers = mesh_data['tri_centers']
      else:
        tri_centers = [-1]
      if dataset_params_list[params_idx].use_fpfh:
        fpfh = mesh_data['mfpfh']
      else:
        fpfh = [-1]
      if dataset_params_list[params_idx].label_per_step:
        labels = mesh_data['labels']
      else:
        labels = mesh_data['label']
      if dataset_params_list[params_idx].kdtree_query_needed:
        kdtree_query = mesh_data['kdtree_query']
      else:
        kdtree_query = [-1]
      if 'dataset_name' in mesh_data.keys():
        name = mesh_data['dataset_name'].tolist() + ':' + fn.decode()
      else:
        name = 'scannet:' + fn.decode()

    yield ([name], vertices, faces, edges, kdtree_query, v_normals, v_rgb, tri_centers, fpfh, labels, [params_idx])

  def __new__(cls, filenames, params_idx):
    return tf.data.Dataset.from_generator(
      cls._generator,
      output_types=cls.OUTPUT_TYPES,
      args=(filenames, params_idx)
    )



class OpenMeshMultiscale(tf.data.Dataset):
  # OUTPUT:      (fn,               vertices_ms,          faces_ms,           edges_ms,           kdtree_query_ms,
  OUTPUT_TYPES = (tf.dtypes.string, tf.dtypes.float32, tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.int32, tf.dtypes.int32,
  # OUTPUT_TYPES = (tf.dtypes.string, tf.dtypes.variant, tf.dtypes.variant, tf.dtypes.variant, tf.dtypes.variant,
              #   ( vertices_normals, vertices_rgb,      faces_centers        fpfh               labels          params_idx)
                  tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.int32, tf.dtypes.int32)

  def _generator(fn_, params_idx):
    fn = fn_[0]
    # TODO: locate all scales of model. if only 1 scale - duplicate it 3 times
    scale_files = glob.glob(fn.decode() + '*.npz')  # fn should be full path + model name
    vertices = []
    faces = []
    edges = []
    labels = []
    kdtree_query = []
    split_ids_vertices = [0]
    split_ids_faces = [0]
    split_ids_edges = [0]
    id_v = 0
    id_f = 0
    id_e = 0
    for f in scale_files:
      with np.load(f, encoding='latin1', allow_pickle=True) as mesh_data:
        vertices.append(mesh_data['vertices'])
        faces.append(mesh_data['faces'])
        edges.append(mesh_data['edges'])
        id_v += mesh_data['vertices'].shape[0]
        split_ids_vertices.append(id_v)
        id_f += mesh_data['faces'].shape[0]
        split_ids_faces.append(id_f)
        id_e += mesh_data['edges'].shape[0]
        split_ids_edges.append(id_e)
        if dataset_params_list[params_idx].label_per_step:
          labels.append(mesh_data['labels'])
        else:
          labels.append(mesh_data['label'])
        if dataset_params_list[params_idx].kdtree_query_needed:
          kdtree_query.append(mesh_data['kdtree_query'])
        else:
          kdtree_query.append([-1])
        if 'dataset_name' in mesh_data.keys():
          name = mesh_data['dataset_name'].tolist() + ':' + fn.decode()
        else:
          name = 'scannet:' + fn.decode()
      # if dataset_params_list[params_idx].v_normals:
      #   v_normals = mesh_data['vertex_normals']
      # else:
    v_normals = [-1]
    v_rgb = [-1]
    tri_centers = [-1]
    fpfh = [-1]

    vertices = np.concatenate(vertices, axis=0)
    faces = np.concatenate(faces, axis=0)
    max_edges = np.max([x.shape[1] for x in edges])
    es = -1 * np.ones((np.sum([x.shape[0] for x in edges]), max_edges), dtype=np.int)
    for i,e in enumerate(edges):
      es[split_ids_edges[i]:split_ids_edges[i+1], :e.shape[1]] = e
    kdtree_query = np.concatenate(kdtree_query, axis=0)
    ids_vs, ids_fs, ids_es = [np.asarray(x) for x in [split_ids_vertices, split_ids_faces, split_ids_edges]]
    # labels = labels[-1]
    yield ([name], vertices, ids_vs, faces, ids_fs, es, ids_es, kdtree_query, v_normals, v_rgb, tri_centers, fpfh, labels[-1], [params_idx])


  def __new__(cls, filenames, params_idx):
    return tf.data.Dataset.from_generator(
      cls._generator,
      output_types=cls.OUTPUT_TYPES,
      args=(filenames, params_idx)
    )



def dump_all_fns_to_file(filenames, params):
  if 'logdir' in params.keys():
    for n in range(10):
      log_fn = params.logdir + '/dataset_files_' + str(n).zfill(2) + '.txt'
      if not os.path.isfile(log_fn):
        try:
          with open(log_fn, 'w') as f:
            for fn in filenames:
              f.write(fn + '\n')
        except:
          pass
        break


def tf_mesh_dataset(params, pathname_expansion, mode=None, size_limit=np.inf, shuffle_size=300,
                    permute_file_names=True, min_max_faces2use=[0, np.inf], data_augmentation={},
                    must_run_on_all=False, max_size_per_class=None, min_dataset_size=64*20, filenames=None):
  params_idx = setup_dataset_params(params, data_augmentation)
  number_of_features = dataset_params_list[params_idx].number_of_features
  params.net_input_dim = number_of_features
  mesh_data_to_walk_features.SET_SEED_WALK = 0
  if not filenames:
    filenames = get_file_names(pathname_expansion, min_max_faces2use)
    if params.classes_indices_to_use is not None:
      filenames = filter_fn_by_class(filenames, params.classes_indices_to_use)
    if max_size_per_class is not None:
      filenames = adjust_fn_list_by_size(filenames, max_size_per_class)

    if permute_file_names:
      filenames = np.random.permutation(filenames)
    else:
      filenames.sort()
      filenames = np.array(filenames)
    if size_limit < len(filenames):
      filenames = filenames[:size_limit]
    if len(filenames) < min_dataset_size:
      filenames = filenames.tolist() * (int(min_dataset_size / len(filenames)) + 1)

  if hasattr(params, 'multiscale') and params.multiscale:
    filenames = np.unique(['_'.join(x.split('_')[:-3]) for x in filenames])  # TODO: this currently works only for modelnet40 file format.
  n_items = len(filenames)



  if mode == 'classification':
    dataset_params_list[params_idx].label_per_step = False
  elif mode == 'semantic_segmentation':
    dataset_params_list[params_idx].label_per_step = True
  elif mode == 'self:triplets':
    dataset_params_list[params_idx].label_per_step = True
  else:
    raise Exception('DS mode ?')

  dump_all_fns_to_file(filenames, params)

  def _open_npz_fn(*args):
    return OpenMeshDataset(args, params_idx)

  def _open_multiscale_npz_fn(*args):
    return OpenMeshMultiscale(args, params_idx)

  ds = tf.data.Dataset.from_tensor_slices(filenames)
  if shuffle_size:
    ds = ds.shuffle(shuffle_size)
  if hasattr(params, 'multiscale') and params.multiscale:
    _open_fn = _open_multiscale_npz_fn
    _gen_walk_fn = generate_ms_walk_py_func
  else:
    _open_fn = _open_npz_fn
    _gen_walk_fn = generate_walk_py_fun

  ds = ds.interleave(_open_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.cache()
  ds = ds.map(_gen_walk_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(params.batch_size)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  # ds = ds.cache()
  return ds, n_items


def tf_scannet_dataset(params, filenames, shuffle_size=300, full_coverage=False, data_augmentation={}):
  params_idx = setup_dataset_params(params, data_augmentation)
  number_of_features = dataset_params_list[params_idx].number_of_features
  params.net_input_dim = number_of_features
  mesh_data_to_walk_features.SET_SEED_WALK = 0
  dataset_params_list[params_idx].label_per_step = True
  dataset_params_list[params_idx].full_coverage = full_coverage
  n_items = 1
  dump_all_fns_to_file([filenames], params)
  def _open_npz_fn(*args):
    return OpenMeshDataset(args, params_idx)

  ds = tf.data.Dataset.from_tensor_slices([filenames])
  if shuffle_size:
    ds = ds.shuffle(shuffle_size)
  ds = ds.interleave(_open_npz_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.cache()
  ds = ds.map(generate_walk_py_fun, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(params.batch_size)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  # ds = ds.cache()
  return ds, n_items


def test_scannet_v2():
  train_npz_list = glob.glob('/home/ran/mesh_walker/datasets/scannet_v2_4cm_crops/*train*.npz')
  out_csv = 'scannet_preprocess/statistics.csv'
  csv_rows = []
  for npz in train_npz_list:
    mesh_data = load_model_from_npz(npz)
    neighbor_indices = mesh_data['edges']
    if np.count_nonzero(neighbor_indices > -1) < 2:
      print('debug - standalong vertex?')
    if np.count_nonzero(neighbor_indices < -1):
      print('debug - train npz with negative edges')
    csv_rows.append([npz, mesh_data['vertices'], mesh_data['faces']])
  val_npz_list = glob.glob('/home/ran/mesh_walker/datasets/scannet_v2_4cm_crops/*val*.npz')
  for npz in val_npz_list:
    mesh_data = load_model_from_npz(npz)
    neighbor_indices = mesh_data['edges']
    if np.count_nonzero(neighbor_indices > -1) < 2:
      print('debug - standalong vertex?')
    if np.count_nonzero(neighbor_indices < -1):
      print('debug - val npz with negative edges')
    csv_rows.append([npz, mesh_data['vertices'], mesh_data['faces']])
  test_npz_list = glob.glob('/home/ran/mesh_walker/datasets/scannet_v2_4cm_crops/*test*.npz')
  for npz in test_npz_list:
    mesh_data = load_model_from_npz(npz)
    neighbor_indices = mesh_data['edges']
    if np.count_nonzero(neighbor_indices < -1):
      print('debug -test npz with negative edges')
    csv_rows.append([npz, mesh_data['vertices'], mesh_data['faces']])

  print('All edges are correct (lead to positive indices or -1')
  import csv
  with open(out_csv, 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerows(csv_rows)


def test_new_dataloader():
  test_npz_list = glob.glob('/home/ran/mesh_walker/datasets/scannet_v2/*val*.npz')
  from params_setting import scannet_params
  dataset_params = scannet_params()
  update_params = {'set_seq_len_by_n_faces': False,
                   'adjust_vertical_model': False,
                   'normalize_model': True,
                   'data_augmentaion_vertices_functions': [],
                   'label_per_step': True,
                   'edges_needed': True,
                   'kdtree_query_needed': False,
                   'full_coverage': True,
                   'vertices_needed': True,
                   'number_of_features': 9,
                   'walk_function':  walks.get_seq_random_walk_euclidean_jumps,
                   'fill_features_functions': [fill_xyz_features, fill_rgb_features, fill_vertex_normals_features]
                   }
  for k,v in update_params.items():
    dataset_params[k] = v
  i = 0
  while True:
    print(i)
    i += 1
    test_npz = np.random.choice(test_npz_list)
    mesh_data = load_model_from_npz(test_npz)
    mesh_data_to_walk_features(mesh_data, dataset_params)


def test_multiscale_dataloader():
  from params_setting import modelnet_params
  dataset_params = modelnet_params('rnn')
  dataset_params.multiscale=True
  update_params = {'set_seq_len_by_n_faces': False,
                   'adjust_vertical_model': False,
                   'normalize_model': True,
                   'data_augmentaion_vertices_functions': [],
                   'label_per_step': True,
                   'edges_needed': True,
                   'kdtree_query_needed': False,
                   'full_coverage': True,
                   'vertices_needed': True,
                   'number_of_features': 9,
                   'walk_function': walks.get_seq_random_walk_euclidean_jumps,
                   'fill_features_functions': [fill_xyz_features, fill_rgb_features, fill_vertex_normals_features]
                   }
  for k,v in update_params.items():
    dataset_params[k] = v
  i = 0
  while True:
    print(i)
    i += 1
    npzs = ['/home/ran/mesh_walker/datasets/modelnet40_reported/test_airplane_0627_simplified_to_1000.npz',
           '/home/ran/mesh_walker/datasets/modelnet40_reported/test_airplane_0627_simplified_to_2000.npz',
           '/home/ran/mesh_walker/datasets/modelnet40_reported/test_airplane_0627_simplified_to_4000.npz']

    mesh_data_to_walk_features(mesh_data, dataset_params)


def test_far_points():
  npz_file= [glob.glob('/home/ran/mesh_walker/datasets/modelnet40_reported/test_{}*4000*'.format(c)) for c in
                           ['airplane_0690']]
  model = load_model_from_npz(npz_file[0][0])
  start_points = get_far_points(model['vertices'], 16)
  print('debug')
  # Visualize model  + big dots for starting vertices
  utils.visualize_model(norm_model(model['vertices'], return_val=True),
                        model['faces'],
                        show_edges=False,
                        opacity=0.4,
                        line_width=0.4,
                        all_colors='seashell',
                        show_vertices=start_points
                        )

if __name__ == '__main__':
  # test_multiscale_dataloader()
  test_far_points()
  # seq_len = 400
  # n_walks_per_model = 2
  # # test_scannet_v2()
  # # test_scannet_v2()
  # # test_npz = '/home/ran/mesh_walker/datasets/modelnet40_meshnet/test_bathtub_0108_312.npz'
  # # test_npz_list = glob.glob('/home/ran/mesh_walker/datasets/modelnet40_1k2k4k_upsample/*.npz')
  # test_npz_list = glob.glob('/home/ran/mesh_walker/datasets/3dFUTURE/train*/*.npz')
  # i= 0
  # while True:
  #   print(i)
  #   i+=1
  #   test_npz = np.random.choice(test_npz_list)
  #   mesh_data = load_model_from_npz(test_npz)
  #   mesh_extra = {}
  #   mesh_extra['n_vertices'] = mesh_data['vertices'].shape[0]
  #   for k,v in mesh_data.items():
  #     mesh_extra[k] = v
  #   # mesh_extra['edges'] = mesh_data['edges']
  #   # mesh_extra['faces'] = mesh_data['faces']
  #   # mesh_extra['angles'] = mesh_data['angles']
  #   # mesh_extra['adj_edges'] = mesh_data['adj_edges']
  #   # mesh_extra['trimesh_vertices'] = mesh_data['trimesh_vertices']
  #   # mesh_extra['face_normals'] = mesh_data['face_normals']
  #   # mesh_extra['vertex_normals'] = mesh_data['vertex_normals']
  #   # mesh_extra['adj_faces'] = mesh_data['adj_faces']
  #   # mesh_extra['tri_centers'] = mesh_data['tri_centers']
  #   # mesh_extra['fpfh'] = mesh_data['fpfh']
  #   mesh_extra['k'] = 10
  #   vertices = mesh_data['vertices']
  #   # walk = walks.get_seq_random_walk_constant_global_jumps(mesh_extra, f0=np.random.choice(mesh_data['vertices'].shape[0]), seq_len=400)
  #   walk = walks.get_seq_random_walk_random_global_jumps_new(mesh_extra,
  #                                                          f0=np.random.choice(mesh_data['vertices'].shape[0]),
  #                                                          seq_len=400)
  #
  #   # walk = walks.get_seq_random_walk_constant_global_jumps(mesh_extra, f0=np.random.choice(mesh_data['vertices'].shape[0]), seq_len=400)
  #
  #   utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], line_width=0.7,
  #                             opacity=0.3,point_size=7, walk=walk[0][0], edge_colors='use_cmap')
  #
  #   features = np.zeros((n_walks_per_model, seq_len, 50), dtype=np.float32)
  #   labels = np.zeros((n_walks_per_model, seq_len), dtype=np.int32)
  #   norm_model(vertices)
  #   f_idx = 0
  #   # f_idx = fill_face_features(features[0], f_idx, mesh_data['vertices'], mesh_extra, walk[0], walk[1], seq_len)
  #   # f_idx = fill_xyz_features(features[0], f_idx, mesh_data['vertices'], mesh_extra, walk[0], walk[1], seq_len)
  #   # f_idx = fill_dxdydz_features(features[0], f_idx, vertices, mesh_extra, walk[0], walk[1], seq_len)
  #   # f_idx = fill_fpf_features(features[0], f_idx, vertices, mesh_extra, walk[0], walk[1], seq_len)
  #   # f_idx = fill_fpfh_features(features[0], f_idx, vertices, mesh_extra, walk[0], walk[1], seq_len)
  #   f_idx = fill_jumps(features[0], f_idx, vertices, mesh_extra, walk[0], walk[1], seq_len)
  #   f_idx = fill_vertex_normals_features(features[0], f_idx, mesh_data['vertices'], mesh_extra, walk[0], walk[1], seq_len)
  #   f_idx = fill_vertex_indices(features[0], f_idx, mesh_data['vertices'], mesh_extra, walk[0], walk[1],
  #                                        seq_len)
  #   if np.min(features[0,:,f_idx-1]) < 0:
  #     print('debug')
  #   else:
  #     print(np.max(features[0,:,f_idx-1]))
  #   # utils.config_gpu(False)
  #   # np.random.seed(1)