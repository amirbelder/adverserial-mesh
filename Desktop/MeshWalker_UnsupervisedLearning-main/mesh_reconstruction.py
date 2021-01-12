import os, shutil, time, copy, glob
from easydict import EasyDict
import json
import platform

import cv2
import numpy as np
import tensorflow as tf
import trimesh, open3d
import pyvista as pv
import scipy
import pylab as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from tqdm import tqdm

import rnn_model
import utils
import dataset
import dataset_prepare

N_ITERS = 100000
DUMP_IMAGES = 100
DUMP_PATH = os.path.expanduser('~') + '/mesh_walker/mesh_reconstruction/'

use_sphere_or_model = "sphere" #"model"

def generate_sphere():
  sphere = trimesh.primitives.Sphere()
  mesh = open3d.geometry.TriangleMesh()
  mesh.vertices = open3d.utility.Vector3dVector(sphere.vertices.copy())
  mesh.triangles = open3d.utility.Vector3iVector(sphere.faces.copy())
  mesh, _, _ = dataset_prepare.remesh(mesh, 500)
  vertices = np.array(mesh.vertices)
  dataset.norm_model.sub_mean_for_data_augmentation = False
  dataset.norm_model(vertices)
  mesh = {'vertices': vertices, 'faces': np.array(mesh.triangles), 'label': 0}
  dataset_prepare.prepare_edges_and_kdtree(mesh)

  return mesh


def dump_mesh(mesh_data, path, cpos, iter):
  window_size = [512, 512]
  p = pv.Plotter(off_screen=1, window_size=(int(window_size[0]), int(window_size[1])))
  faces = np.hstack([[3] + f.tolist() for f in mesh_data['faces']])
  surf = pv.PolyData(mesh_data['vertices'], faces)
  p.add_mesh(surf, show_edges=False, color=None)
  p.camera_position = cpos
  p.set_background("#AAAAAA", top="White")
  rendered = p.screenshot()
  p.close()
  img = rendered.copy()
  cv2.putText(img, str(iter), (img.shape[1] - 100, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
              color=(0, 255, 255), thickness=2)
  if not os.path.isdir(path):
    os.makedirs(path)
  cv2.imwrite(path + '/img_' + str(dump_mesh.i).zfill(5) + '.jpg', img)
  dump_mesh.i += 1
dump_mesh.i = 0


def calc_ftr_vector(params, dnn_model, npz_fn):
  params = copy.deepcopy(params)
  params.n_walks_per_model = 16
  ds, n_models = dataset.tf_mesh_dataset(params, pathname_expansion=npz_fn, mode=params.network_task,
                                         must_run_on_all=True)
  for data in ds:
    name, ftrs, gt = data
    predictions = dnn_model(ftrs[0, :, :, :3], classify=False, training=False)
    break

  feature_vector = tf.reduce_mean(predictions, axis=0)

  return feature_vector

def set_hyper_params():
  #move all to a yaml file?
  hyper_params = {}
  learning_weight = 0.01
  max_label_diff = 0.1
  max_iter = 10000
  target_label = 15
  source_label = 25
  iter_count = 0
  #loss =
  #attack
  #mesh
  last_pred = -1
  N_ITERS = 100000
  DUMP_IMAGES = 100
  DUMP_PATH = os.path.expanduser('~') + '/mesh_walker/mesh_reconstruction/'

  use_sphere_or_model = "model"

def mesh_reconstruction(logdir, dataset_folder):
  with open(logdir + '/params.txt') as fp:
    params = EasyDict(json.load(fp))
  model_fn = tf.train.latest_checkpoint(logdir)
  params.batch_size = 1
  #params.seq_len = 400
  params.n_walks_per_model = 1
  params.set_seq_len_by_n_faces = False
  params.data_augmentaion_vertices_functions = []
  params.label_per_step = False
  params.n_target_vrt_to_norm_walk = 0
  params.net_input += ['vertex_indices']
  dataset.setup_features_params(params, params)
  dataset.mesh_data_to_walk_features.SET_SEED_WALK = False

  dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, 3, model_fn,
                                   model_must_be_load=True, dump_model_visualization=False)

  #Amir
  # Here I can put the right npz file that holds a mesh with a source Domain mesh
  npz_fn = "datasets_processed/shrec11/16-04_a/train/T187_not_changed_500.npz"
  #os.path.expanduser('~') + '/mesh_walker/datasets_processed/shrec11/16-04_a/test/T10_not_changed_500.npz'
  if use_sphere_or_model == "model":
    orig_mesh_data = np.load(npz_fn, encoding='latin1', allow_pickle=True)
    mesh_data = {k: v for k, v in orig_mesh_data.items()}
    #mesh_data['vertices'] += np.random.normal(size=mesh_data['vertices'].shape) * .02

  elif use_sphere_or_model == "sphere":
    mesh_data = generate_sphere()

  #utils.visualize_model(orig_mesh_data['vertices'], orig_mesh_data['faces'])
  #utils.visualize_model(mesh_data['vertices'], mesh_data['faces'])

  #target_feature_vector = calc_ftr_vector(params, dnn_model, npz_fn)
  target_label = 15
  target_feature_vector = tf.one_hot(target_label, 30)

  skip = int(params.seq_len / 2)
  if os.path.isdir(DUMP_PATH):
    shutil.rmtree(DUMP_PATH)

  l = []
  cpos = None
  w = 0.04
  for n in range(N_ITERS):
    if n == int(N_ITERS / 2):
      w = w / 1.5
    features, labels = dataset.mesh_data_to_walk_features(mesh_data, params)

    ftrs = tf.cast(features[:, :, :3], tf.float32)
    v_indices = features[0, :, 3].astype(np.int)

    with tf.GradientTape() as tape:
      tape.watch(ftrs)
      pred = dnn_model(ftrs, classify=True, training=False)
      #loss = w * tf.keras.losses.sparse_categorical_crossentropy(target_label, pred)    # 18 = two_balls , 15 - horse
      loss = w * tf.keras.losses.mean_squared_error(target_feature_vector, pred[0])

    gradients = tape.gradient(loss, ftrs)
    print(n, loss.numpy(), np.argmax(pred))
    l.append(loss.numpy())

    mesh_data['vertices'][v_indices[skip:]] -= gradients[0][skip:].numpy()

    if DUMP_IMAGES and n % DUMP_IMAGES == 0:
      cpos = dump_mesh(mesh_data, DUMP_PATH, cpos, n)

  if 1:
    plt.plot(l)
    plt.show()
    #utils.visualize_model(mesh_data['vertices'], mesh_data['faces'])

  cmd = f'ffmpeg -framerate 24 -i {DUMP_PATH}img_%05d.jpg {DUMP_PATH}mesh_reconstruction.mp4'
  os.system(cmd)


def main():
  np.random.seed(0)
  utils.config_gpu(1)

  iter2use = 'last'
  classes_indices_to_use = None
  model_fn = None

<<<<<<< HEAD
  logdir = "../../mesh_walker/runs_aug_360_must/0078-06.01.2021..15.42__camel_horse_xyz__shrec11_16-04_a/" #'/home/alonla/mesh_walker/runs_aug_360_must/0004-11.09.2020..04.35__shrec11_16-04_A'
  dataset_path = 'datasets_processed/shrec11/16-04_a/test/*.npz' #os.path.expanduser('~') + '/mesh_walker/datasets_processed/shrec11/16-04_a/test/*.npz'
=======
  logdir = "../../mesh_walker/runs_aug_360_must/0048-30.12.2020..17.19__shrec11_16-04_a"  # '/home/alonla/mesh_walker/runs_aug_360_must/0004-11.09.2020..04.35__shrec11_16-04_A'
  dataset_path = 'datasets_processed/shrec11/16-04_a/test/*.npz'  # os.path.expanduser('~') + '/mesh_walker/datasets_processed/shrec11/16-04_a/test/*.npz'
>>>>>>> 7337e131b97494242dfa1fa2d6d51862e37718c3
  if 0:
    acc, _ = evaluate_classification.calc_accuracy_test(logdir=logdir,
                                                        dataset_folder=dataset_path,
                                                        labels=dataset_prepare.shrec11_labels, iter2use=iter2use,
                                                        n_walks_per_model=8)
    print(acc)
  else:
    mesh_reconstruction(logdir=logdir, dataset_folder=dataset_path)

  return 0

if __name__ == '__main__':
  main()