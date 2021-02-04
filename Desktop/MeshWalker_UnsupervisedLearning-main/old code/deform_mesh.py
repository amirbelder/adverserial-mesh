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
import yaml
import argparse

import rnn_model
import utils
import dataset
import dataset_prepare
import evaluate_clustering


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


def dump_mesh(mesh_data, path, cpos, iter, target_pred = -1, source_pred = -1):
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
  if target_pred != -1 and source_pred != -1: #both are none default vales
    my_text = str(iter) + " " + "source pred: " + str(source_pred) + " target pred: " + str(target_pred)
  else:
    my_text = str(iter)
  cv2.putText(img, my_text, (img.shape[1] - 100, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
              color=(0, 255, 255), thickness=2)
  if not os.path.isdir(path):
    os.makedirs(path)
  cv2.imwrite(path + '/img_' + str(dump_mesh.i).zfill(5) + '.jpg', img)
  dump_mesh.i += 1
dump_mesh.i = 0

def dump_prec(mesh_data, path, cpos, iter, target_pred = -1, source_pred = -1):
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
  if target_pred != -1 and source_pred != -1:  # both are none default vales
    my_text = str(iter) + " " + "source pred: " + str(source_pred) + " target pred: " + str(target_pred)
  else:
    my_text = str(iter)
  cv2.putText(img, my_text, (img.shape[1] - 100, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
              color=(0, 255, 255), thickness=2)
  if not os.path.isdir(path):
    os.makedirs(path)
  prec = str(source_pred)
  cv2.imwrite(path + '/img_' + prec + "_" + str(dump_mesh.i).zfill(5) + '.jpg', img)
  dump_mesh.i += 1
dump_mesh.i = 0

def deform_add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, dump_model=True):
  m = {}
  for k, v in mesh_data.items():
    if k in fileds_needed:
      m[k] = v
  for field in fileds_needed:
    if field not in m.keys():
      if field == 'labels_fuzzy':
        m[field] = np.zeros((0,))
      if field == 'walk_cache':
        m[field] = np.zeros((0,))
      if field == 'kdtree_query':
        dataset_prepare.prepare_edges_and_kdtree(m)

  if dump_model:
    np.savez(out_fn, **m)


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


def get_res_path(config):
  shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
  ]

  res_path = '../../mesh_walker/' + shrec11_labels[config['source_label']] + '_to_' + shrec11_labels[config['target_label']]
  return res_path

def get_mesh_path_500(config):
  if config['source_label'] == 1:
    return 'datasets_processed/shrec11/16-04_a/train/T437_not_changed_500.npz' #man
  if config['source_label'] == 4:
    return 'datasets_processed/shrec11/16-04_a/train/T504_not_changed_500.npz' #dog2
  if config['source_label'] == 7:
    return 'datasets_processed/shrec11/16-04_a/train/T136_not_changed_500.npz' #dog1
  if config['source_label'] == 9:
    #return 'datasets_processed/shrec11/16-04_a/train/T520_not_changed_500.npz' #bird2
    return 'datasets_processed/shrec11/16-04_a/train/T358_not_changed_500.npz' # bird2 upsidedown
  if config['source_label'] == 15:
    return 'datasets_processed/shrec11/16-04_a/train/T295_not_changed_500.npz' # This is a horse
  if config['source_label'] == 23:
    return 'datasets_processed/shrec11/16-04_a/train/T36_not_changed_500.npz' # woman
  if config['source_label'] == 25:
    return 'datasets_processed/shrec11/16-04_a/train/T398_not_changed_500.npz' #camel
  if config['source_label'] == 29:
    return 'datasets_processed/shrec11/16-04_a/train/T60_not_changed_500.npz' #bird1
  return '../../mesh_walker/man_to_man/last_model.npz'
  #return None

def mesh_reconstruction(config):
  with open(config['trained_model'] + '/params.txt') as fp:
    params = EasyDict(json.load(fp))
  model_fn = tf.train.latest_checkpoint(config['trained_model'])
  params.batch_size = 1
  params.seq_len = config['walk_len']
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

  prec_arr = [x * 0.1 for x in range(11)]
  was_made = [False for i in range(10)]
  result_path=get_res_path(config)
  if os.path.isdir(result_path):
      shutil.rmtree(result_path)

  if config['use_last'] is True and os.listdir(result_path).__contains__('last_model.npz'):
    orig_mesh_data = np.load(result_path + '/last_model.npz', encoding='latin1', allow_pickle=True)
    mesh_data = {k: v for k, v in orig_mesh_data.items()}
  else:
      # Here I can put the right npz file that holds a mesh with a source Domain mesh
    if config['use_sphere_or_model'] == 'model':
      mesh_path = get_mesh_path_500(config=config)
      orig_mesh_data = np.load(mesh_path, encoding='latin1', allow_pickle=True)
      mesh_data = {k: v for k, v in orig_mesh_data.items()}
        #mesh_data['vertices'] += np.random.normal(size=mesh_data['vertices'].shape) * .02

    elif config['use_sphere_or_model'] == 'sphere':
      mesh_data = generate_sphere()

  #utils.visualize_model(orig_mesh_data['vertices'], orig_mesh_data['faces'])
  #utils.visualize_model(mesh_data['vertices'], mesh_data['faces'])

  #target_feature_vector = calc_ftr_vector(params, dnn_model, npz_fn)
  target_feature_vector = tf.one_hot(config['target_label'], 30)

  skip = int(params.seq_len / 2)

  l = []
  cpos = None
  w = config['learning_weight']
  teta = config['max_label_diff']
  num_iter_no_change = 0
  last_dev_res = 0
  skipped_iters = 0
  res_path = get_res_path(config)
  fields_needed = ['vertices', 'faces', 'edges', 'kdtree_query', 'label', 'labels', 'dataset_name', 'labels_fuzzy']

  for num_iter in range(config['max_iter']):
    features, labels = dataset.mesh_data_to_walk_features(mesh_data, params)
    ftrs = tf.cast(features[:, :, :3], tf.float32)
    v_indices = features[0, :, 3].astype(np.int)

    with tf.GradientTape() as tape:
      tape.watch(ftrs)
      pred = dnn_model(ftrs, classify=True, training=False)
      if config['attack'] == 'sparse_categorical_crossentropy':
        #loss = w * tf.keras.losses.sparse_categorical_crossentropy(target_feature_vector, pred[0])    # 18 = two_balls , 15 - horse
        loss = w * tf.keras.losses.KLDivergence(target_feature_vector, pred[0])
      else: # default
        loss = w * tf.keras.losses.mean_squared_error(target_feature_vector, pred[0])
        #print(pred[0])
    if num_iter == 0:
      prev_target_pred = (pred[0].numpy())[config['target_label']]
      prev_source_pred = (pred[0].numpy())[config['source_label']]

    attack = loss

    #if tf.abs(tf.math.unsorted_segment_sum(tf.math.subtract(last_pred, pred[0]))) > teta:
    #   attack * = teta

    # If we got this walk right, we might not want to upate this part

    if np.argmax(pred) == config['target_label']:
      skipped_iters += 1
      if skipped_iters < 10:
        print("skipped iter: ", num_iter)
        continue
      else:
        skipped_iters = 0

    # Check to see that we didn't update too much
    # this is: pred(source_label) - pred(target_label)
    #last_pred_diff = (pred[0].numpy())[config['source_label']] - (pred[0].numpy())[config['target_label']]

    target_pred = (pred[0].numpy())[config['target_label']]
    source_pred = (pred[0].numpy())[config['source_label']]
    if abs(prev_source_pred - source_pred) < config['no_change_threshold'] or abs(prev_target_pred - target_pred) < config['no_change_threshold']:
      num_iter_no_change+=1

    if num_iter_no_change > config['iterations_with_no_changes']:
      if w * 2 < config['weight_threshold']:
        w *= 2
      num_iter_no_change=0

    if abs(prev_source_pred - source_pred) > teta or abs(prev_target_pred - target_pred) > teta:
      if w > 10e-3:
        w /= 2

    if num_iter % config['iter_2_change_weight'] == 0:
      if w * 10 < config['weight_threshold']:
        w *= 10

    gradients = tape.gradient(attack, ftrs)
    gpu_tmpr = utils.get_gpu_temprature()
    print("iter:", num_iter, " attack:", attack.numpy(), " w:", w, " target prec:",
          (pred[0].numpy())[config['target_label']], " source prec:", (pred[0].numpy())[config['source_label']], " max label:", np.argmax(pred), " gpu_temp: ", gpu_tmpr)
    l.append(loss.numpy())

    mesh_data['vertices'][v_indices[skip:]] += gradients[0][skip:].numpy()
    # Updating the last prediction
    prev_target_pred = target_pred
    prev_source_pred = source_pred

    curr_iter = num_iter - (num_iter % config['image_save_iter'])
    if curr_iter / config['image_save_iter'] >= last_dev_res + 1 or num_iter==0:
      cpos = dump_mesh(mesh_data, result_path, cpos, num_iter, target_pred, source_pred)
      last_dev_res = num_iter / config['image_save_iter']
      deform_add_fields_and_dump_model(mesh_data=mesh_data, fileds_needed=fields_needed, out_fn=result_path + '/last_model.npz')#"+ str(num_iter))

    """curr_prec_range_index = int(source_pred*10)
    if prec_arr[curr_prec_range_index] < source_pred <prec_arr[curr_prec_range_index+1] and was_made[curr_prec_range_index] is False:
      dump_prec(mesh_data, result_path + '/percent/', cpos, num_iter, target_pred, source_pred)
      was_made[curr_prec_range_index] = False

    if num_iter % config['image_save_iter'] == 0:
      cpos = dump_mesh(mesh_data, config['result_path'], cpos, num_iter)"""

    if config['show_model_every'] > 0 and num_iter % config['show_model_every'] == 0:
      plt.plot(l)
      plt.show()
      utils.visualize_model(mesh_data['vertices'], mesh_data['faces'])

  #res_path = config['result_path']
  cmd = f'ffmpeg -framerate 24 -i {res_path}img_%05d.jpg {res_path}mesh_reconstruction.mp4'
  os.system(cmd)
  return

def check_model_accuracy():
  iter2use = 'last'
  classes_indices_to_use = None
  model_fn = None

  logdir = "../../mesh_walker/runs_aug_360_must/0078-06.01.2021..15.42__camel_horse_xyz__shrec11_16-04_a/"  # '/home/alonla/mesh_walker/runs_aug_360_must/0004-11.09.2020..04.35__shrec11_16-04_A'
  dataset_path = 'datasets_processed/shrec11/16-04_a/test/*.npz'  # os.path.expanduser('~') + '/mesh_walker/datasets_processed/shrec11/16-04_a/test/*.npz'


  acc, _ = evaluate_clustering.calc_accuracy_test(logdir=logdir,
                                                        dataset_folder=dataset_path,
                                                        labels=dataset_prepare.shrec11_labels, iter2use=iter2use,
                                                        n_walks_per_model=8)
  return

def main():
  np.random.seed(0)
  utils.config_gpu(1, -1)

  #get hyper params from yaml
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default='recon_config.yaml', help='Path to the config file.')
  opts = parser.parse_args()
  config = utils.get_config(opts.config)

  #check_model_accuracy()
  print("source label: ", config['source_label'], " target label: ", config['target_label'], " output dir: ", get_res_path(config))
  mesh_reconstruction(config)

  return 0

if __name__ == '__main__':
  main()
