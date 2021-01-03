
#!!!!!!! NOT TO BE INCLUDEDE !!!!!#

import os
from tqdm import tqdm

import trimesh
import open3d
import numpy as np
import tensorflow as tf

shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
]
shrec11_shape2label = {v: k for k, v in enumerate(shrec11_labels)}


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_example(mesh_data):
    feature = {
        'name':             _bytes_feature(str.encode(mesh_data['name'])),
        'num_vertices':     _int64_feature(mesh_data['vertices'].shape[0]),
        'num_triangles':    _int64_feature(mesh_data['triangles'].shape[0]),
        'vertices':         tf.train.Feature(float_list=tf.train.FloatList(value=mesh_data['vertices'].flatten())),
        'triangles':        tf.train.Feature(int64_list=tf.train.Int64List(value=mesh_data['triangles'].flatten())),
        'neighbors':        tf.train.Feature(int64_list=tf.train.Int64List(value=mesh_data['neighbors'].flatten())),
    }
    optional_keys = ['mesh_label', 'vertex_labels', 'triangle_labels', 'edge_labels']
    for key in optional_keys:
      if key in mesh_data:
        d = mesh_data[key]
        if type(d) is type(np.array([])):
          feature[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=d.flatten()))
        else:
          feature[key] = _int64_feature(d)

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

    return tf_example


def faces2neighbors(faces, n_vertices, MAX_NEIGHBORS=16):
  def _add_nbr(e0, e1):
    if e0 != e1 and e1 not in neighbors[e0]:
      assert n_nbrs[e0] < MAX_NEIGHBORS, 'number of neighbors must be less than ' + str(MAX_NEIGHBORS)
      neighbors[e0, n_nbrs[e0]] = e1
      n_nbrs[e0] += 1

  neighbors = -np.ones((n_vertices, MAX_NEIGHBORS), dtype=np.int32)
  n_nbrs = np.zeros((n_vertices,), dtype=np.int32)
  for f in faces:
    for edge in [[f[0], f[1]], [f[1], f[2]], [f[2], f[0]]]:
      _add_nbr(edge[0], edge[1])
      _add_nbr(edge[1], edge[0])

  return neighbors


def remesh(mesh_orig, target_n_faces, labels_orig=None):
  labels = labels_orig
  if target_n_faces < np.asarray(mesh_orig.triangles).shape[0]:
    mesh = mesh_orig.simplify_quadric_decimation(target_n_faces)
    mesh = mesh.remove_unreferenced_vertices()
    if labels_orig:
      labels = fix_labels_by_dist(np.asarray(mesh.vertices), np.asarray(mesh_orig.vertices), labels_orig)
  else:
    mesh = mesh_orig

  return mesh, labels


def get_mesh_data(mesh_fn, label, dataset_config):
  """get mesh data"""
  if dataset_config['classification']:
    mesh_ = trimesh.load_mesh(mesh_fn, process=True)
    mesh_.remove_duplicate_faces()
  else:
    mesh_ = trimesh.load_mesh(mesh_fn, process=False)
  mesh = open3d.geometry.TriangleMesh()
  mesh.vertices = open3d.utility.Vector3dVector(mesh_.vertices)
  mesh.triangles = open3d.utility.Vector3iVector(mesh_.faces)

  mesh, _ = remesh(mesh, dataset_config['mesh_simplification_to_#faces'])

  vertices = np.asarray(mesh.vertices).astype(np.float32)
  triangles = np.asarray(mesh.triangles).astype(np.int32)
  neighbors = faces2neighbors(triangles, vertices.shape[0])
  vertex_labels = None

  res = {
    'name': mesh_fn,
    'num_vertices': np.array([vertices.shape[0]]),
    'num_triangles': np.array([triangles.shape[0]]),
    'vertices': vertices,
    'triangles': triangles,
    'neighbors': neighbors,
    'mesh_label': np.array([label]),
    'vertex_labels': vertex_labels,
  }

  return res


def no_overlap_fns(train_meshes, test_meshes):
  """make sure no overlap between train meshes and test ones"""
  all = set()
  for meshes in [train_meshes, test_meshes]:
    for m in meshes:
      all.add(os.path.realpath(m.lower()))

  return len(all) == len(train_meshes) + len(test_meshes)


def get_meshes_names_shrec11_raw(dataset_path, dataset_config):
  def _get_mesh_fn_list(cat_ids2use):
    return [dataset_path + '/raw/T' + str(id) + '.off' for id in cat_ids2use]
  # Prepare labels per model name
  current_label = None
  model_number2label = [-1 for _ in range(600)]
  for line in open(dataset_path + '/evaluation/test.cla'):
    sp_line = line.split(' ')
    if len(sp_line) == 3:
      name = sp_line[0].replace('_test', '')
      if name in shrec11_labels:
        current_label = name
      else:
        raise Exception('?')
    if len(sp_line) == 1 and sp_line[0] != '\n':
      model_number2label[int(sp_line[0])] = shrec11_shape2label[current_label]

  n_train_examples = int(dataset_config['split'].split('-')[0])
  n_test_examples = 20 - n_train_examples
  model_number2label = np.array(model_number2label)
  train_meshes = []
  train_labels = []
  test_meshes = []
  test_labels = []
  for cat in range(30):
    cat_ids = np.where(model_number2label == cat)[0]
    assert cat_ids.size == 20
    cat_ids = np.random.permutation(cat_ids)
    train_meshes += _get_mesh_fn_list(cat_ids[:n_train_examples])
    test_meshes += _get_mesh_fn_list(cat_ids[n_train_examples:])
    train_labels += [cat] * n_train_examples
    test_labels += [cat] * n_test_examples

  assert no_overlap_fns(train_meshes, test_meshes), 'train set and test set must not share meshes!'

  return {'train': {'meshes': train_meshes, 'labels': train_labels},
          'test':  {'meshes': test_meshes,  'labels': test_labels }}


def get_meshes_names(dataset_path, dataset_name, dataset_config):
  """according to the dataset type & input directory, generate 2 lists of mesh fine names:
  one for the train part and one for the test part.
  """
  if dataset_config['type'] == 'shrec11_raw':
    return get_meshes_names_shrec11_raw(dataset_path, dataset_config)


def dict_to_example(instance):
  feature = {}
  for key, value in instance.items():
    if value is None:
      feature[key] = tf.train.Feature()
    elif type(value) is str:
      feature[key] = _bytes_feature(str.encode(value))
    elif value.dtype == np.integer or value.dtype == np.int32:
      feature[key] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=value.flatten().tolist()))
    elif value.dtype == np.float32:
      feature[key] = tf.train.Feature(
          float_list=tf.train.FloatList(value=value.flatten().tolist()))
    else:
      feature[key] = tf.train.Feature(
          bytes_list=tf.train.BytesList(value=value.flatten().tolist()))

  return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_tfrecords(dataset_name, dataset_config, dataset_path, output_dir):
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
  meshes_names_and_labels = get_meshes_names(dataset_path, dataset_name, dataset_config)
  for mode, data in meshes_names_and_labels.items():
    writer_name = dataset_name + '_' + dataset_config['split'] + '_' + mode + '.tfrecord'
    with tf.io.TFRecordWriter(os.path.join(output_dir, writer_name)) as writer:
      for mesh_fn, label in tqdm(zip(data['meshes'], data['labels']), total=len(data['labels'])):
        mesh_data = get_mesh_data(mesh_fn, label, dataset_config)
        mesh_example = dict_to_example(mesh_data)
        writer.write(mesh_example.SerializeToString())


if __name__ == '__main__':
  np.random.seed(0)
  dataset_name = 'shrec11'
  path = '/home/alonlahav/datasets_processed/shrec11/'
  dataset_config = {'split': '10-10',
                    'type': 'shrec11_raw',
                    'mesh_simplification_to_#faces': 1000,
                    'classification': True,
                    }
  dataset_path = '/home/alonlahav/datasets_processed/shrec11/'
  output_dir = '/home/alonlahav/mesh_walker/datasets_processed-tmp/tfrecords'

  generate_tfrecords(dataset_name, dataset_config, dataset_path, output_dir)
