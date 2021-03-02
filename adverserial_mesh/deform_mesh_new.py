import argparse
import utils
#get hyper params from yaml
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='recon_config.yaml', help='Path to the config file.')
opts = parser.parse_args()
config = utils.get_config(opts.config)

if config['gpu_to_use'] >= 0:
  utils.set_single_gpu(config['gpu_to_use'])


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

def add_spheres_as_2_balls_in_dataset(dataset_path = '', mode = 'train'):
  # mode can be train or test and will effect the range of the loop

  if dataset_path == '':
    return
  meshs = []
  two_balls_idx = 18
  classes_indices_to_use = [two_balls_idx]
  two_balls_filenames = None

  i = 0 if mode == 'train' else 16
  max_num = 16 if mode == 'train' else 20

  while i < max_num:
    new_mesh = generate_sphere(radius=1.0+0.5*i, label=two_balls_idx)
    meshs.append(new_mesh)
    i+=1

  filenames = dataset.get_file_names(dataset_path, [0, np.inf])

  if classes_indices_to_use is not None:
    two_balls_filenames = dataset.filter_fn_by_class(filenames, classes_indices_to_use)

  if two_balls_filenames is not None:
    for idx,file in enumerate(two_balls_filenames):
      two_balls_mesh = np.load(file, encoding='latin1', allow_pickle=True)
      mesh_data = {k: v for k, v in two_balls_mesh.items()}
      mesh_data['vertices'] = meshs[idx]['vertices']
      mesh_data['faces'] = meshs[idx]['faces']
      np.savez(file, **mesh_data)

  return

def generate_sphere(radius = 1.0, label = 0):
  sphere = trimesh.primitives.Sphere(radius = radius)
  mesh = open3d.geometry.TriangleMesh()
  mesh.vertices = open3d.utility.Vector3dVector(sphere.vertices.copy())
  mesh.triangles = open3d.utility.Vector3iVector(sphere.faces.copy())
  mesh, _, _ = dataset_prepare.remesh(mesh, 500)
  vertices = np.array(mesh.vertices)
  dataset.norm_model.sub_mean_for_data_augmentation = False
  dataset.norm_model(vertices)
  mesh = {'vertices': vertices, 'faces': np.array(mesh.triangles), 'label': label}
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


def change_one_tarj(config):
  print("change 1 traj: ", str(config['source_label']), " to ", str(config['target_label']))
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

  result_path, _ = get_res_path(config)
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

    elif config['use_sphere_or_model'] == 'sphere':
      mesh_data = generate_sphere()

  target_feature_vector = tf.one_hot(config['target_label'], 30)

  w = config['learning_weight']
  kl_divergence_loss = tf.keras.losses.KLDivergence()

  for num_iter in range(config['max_iter']):
    features, labels = dataset.mesh_data_to_walk_features(mesh_data, params)
    ftrs = tf.cast(features[:, :, :3], tf.float32)

    with tf.GradientTape() as tape:
      tape.watch(ftrs)
      pred = dnn_model(ftrs, classify=True, training=False)

      # Produce the attack
      sparse_attack = 0 if config['sparse_alpha'] == 0 else w * tf.keras.losses.sparse_categorical_crossentropy(config['target_label'], pred)
      kl_div_attack = 0 if config['kl_div_alpha'] == 0 else w * kl_divergence_loss(config['target_label'], pred)
      mse_attack = 0 if config['mse_alpha'] == 0 else w * tf.keras.losses.mean_squared_error(target_feature_vector, pred[0])
      # No chamfer dist for now, as it's supposed to be between 2 sets of points, and not preds

      attack = config['sparse_alpha'] * sparse_attack + config['kl_div_alpha'] * kl_div_attack + config['mse_alpha'] * mse_attack

    gradients = tape.gradient(attack, ftrs)

    is_close_enough_to_target = (pred[0].numpy())[config['target_label']] > config['pred_close_enough_to_target']
    counter = 0
    while not is_close_enough_to_target:
      if counter == 0:
        print("new traj, iter is", num_iter)
        ftrs_after_attack_update = ftrs
      ftrs_after_attack_update -= gradients
      ftrs_after_attack_update -= gradients
      new_pred_pos = dnn_model(ftrs_after_attack_update + gradients, classify=True, training=False)
      target_pred_pos = (new_pred_pos[0].numpy())[config['target_label']]
      new_pred_neg = dnn_model(ftrs_after_attack_update - gradients, classify=True, training=False)
      source_pred_pos = (new_pred_pos[0].numpy())[config['source_label']]
      if target_pred_pos > source_pred_pos:
        new_pred = new_pred_pos
      else:
        new_pred = new_pred_neg
      target_pred = (new_pred[0].numpy())[config['target_label']]
      is_close_enough_to_target = target_pred > config['pred_close_enough_to_target']
      counter+=1
      if counter > 1000:
        print("Failed")
        is_close_enough_to_target=True

    print("traj took: ", counter, " iterations to change to 100%")

def get_res_path(config):
  shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
  ]

  #fn = npz_fn.split('/')[-1].split('.')[-2]
  net_name = config['trained_model'].split('/')[-1]
  if len(net_name) > 0:
    res_path = '../mesh_walker/attacks/' +net_name +'/' + shrec11_labels[config['source_label']] + '_to_' + shrec11_labels[config['target_label']]
  else:
    res_path = '../mesh_walker/attacks/'  + shrec11_labels[config['source_label']] + '_to_' + shrec11_labels[config['target_label']]
  return res_path, net_name

def get_mesh_path_500(config):
  if config['source_label'] == 1:
    return 'datasets_processed/shrec11/16-04_a/train/T437_not_changed_500.npz' #man
  if config['source_label'] == 4:
    #return 'datasets_processed/shrec11/16-04_a/train/T504_not_changed_500.npz' #dog2
    return 'datasets_processed/shrec11/16-04_a/train/T93_not_changed_500.npz' # A green dog2
  if config['source_label'] == 7:
    #return 'datasets_processed/shrec11/16-04_a/train/T136_not_changed_500.npz' #dog1
    return 'datasets_processed/shrec11/16-04_a/train/T197_not_changed_500.npz' # A green dog1
  if config['source_label'] == 9:
    #return 'datasets_processed/shrec11/16-04_a/train/T520_not_changed_500.npz' #bird2
    return 'datasets_processed/shrec11/16-04_a/train/T358_not_changed_500.npz' # bird2 upsidedown
  if config['source_label'] == 15:
    #return 'datasets_processed/shrec11/16-04_a/train/T295_not_changed_500.npz' # This is a horse
    return 'datasets_processed/shrec11/16-04_a/train/T234_not_changed_500.npz' # A green horse
  if config['source_label'] == 23:
    return 'datasets_processed/shrec11/16-04_a/train/T36_not_changed_500.npz' # woman
  if config['source_label'] == 25:
    #return 'datasets_processed/shrec11/16-04_a/train/T398_not_changed_500.npz' #camel
    return 'datasets_processed/shrec11/16-04_a/train/T497_not_changed_500.npz' # A green camel
  if config['source_label'] == 29:
    return 'datasets_processed/shrec11/16-04_a/train/T60_not_changed_500.npz' #bird1
  return '../../mesh_walker/man_to_man/last_model.npz'
  #return None


def plot_preditions(params, dnn_model, config, mesh_data, result_path, num_iter, x_axis, source_pred_list, target_pred_list, predictions):
  params.n_walks_per_model = 16
  features, labels = dataset.mesh_data_to_walk_features(mesh_data, params)
  ftrs = tf.cast(features[:, :, :3], tf.float32)
  eight_pred = dnn_model(ftrs, classify=True, training=False)
  sum_pred = tf.reduce_sum(eight_pred, 0)
  print("target_label number ", config['target_label'],  " over " + str(params.n_walks_per_model) + " runs is: ", (sum_pred.numpy())[config['target_label']] / params.n_walks_per_model)
  print("source_label number ", config['source_label'],  " over " + str(params.n_walks_per_model) + " runs is: ", (sum_pred.numpy())[config['source_label']] / params.n_walks_per_model)
  source_pred_list.append((sum_pred.numpy())[config['source_label']] / params.n_walks_per_model)
  target_pred_list.append((sum_pred.numpy())[config['target_label']] / params.n_walks_per_model)
  for i in range(len(predictions)):
    predictions[i].append((sum_pred.numpy())[config['classes_to_train'][i]] / params.n_walks_per_model)
  params.n_walks_per_model = 8

  if not os.path.isdir(result_path + '/plots/'):
    os.makedirs(result_path + '/plots/')
  # plot the predictions
  x_axis.append(num_iter)

  for i in range(len(predictions)):
    plt.plot(x_axis, predictions[i])
    name_of_graph = str(config['classes_to_train'][i]) + ' prediction'
    plt.title(name_of_graph)
    plt.savefig(result_path + '/plots/' + name_of_graph +'.png')
    plt.close()

  plt.plot(x_axis, source_pred_list)
  plt.title(str(config['source_label']) + " to " + str(config['target_label']) + ": source pred")
  plt.savefig(result_path + '/plots/' + 'source_pred.png')
  plt.close()

  if config['source_label'] != config['target_label']:
    plt.plot(x_axis, target_pred_list)
    plt.title(str(config['source_label']) + " to " + str(config['target_label']) + ": target pred")
    plt.savefig(result_path + '/plots/' + 'target_pred.png')
    plt.close()
  return

def set_attack(config, away_from_source_attack, forward_target_attack, num_iter):
  """
  Returns the attack according to the kind we need,  either away from the source or forward the target
  """
  if config['attack_direction'] == 'both':
    if num_iter % 2 == 0:
        return away_from_source_attack * -1.0
    else:
        return forward_target_attack
  elif config['attack_direction'] == 'away':
    return away_from_source_attack * -1.0
  elif config['attack_direction'] == 'forward':
    return forward_target_attack
  elif config['attack_direction'] == 'after':
    if num_iter > config['thrs_iters_for_forward']:
      return forward_target_attack
    else:
      return away_from_source_attack * -1.0
  else:
    return 0


def chose_target(config, dnn_model, params, mesh_path = None):
  """
  We will set the target to be the class with the second best prediction
  """
  if mesh_path is None:
    mesh_path = get_mesh_path_500(config=config)
  orig_mesh_data = np.load(mesh_path, encoding='latin1', allow_pickle=True)
  mesh_data = {k: v for k, v in orig_mesh_data.items()}

  features, labels = dataset.mesh_data_to_walk_features(mesh_data, params)
  ftrs = tf.cast(features[:, :, :3], tf.float32)
  pred = dnn_model(ftrs, classify=True, training=False)
  pred = tf.reduce_sum(pred, 0)
  pred /= params.n_walks_per_model
  sorted_preds = np.argsort(pred)
  # Setting the class with the second best prediction as the target
  if config['source_label'] == sorted_preds[-1]:
    config['target_label'] = sorted_preds[-2]
  elif config['source_label'] == sorted_preds[-2]:
    config['target_label'] = sorted_preds[-1]
  #else:
  #  print("Bas choise of model! The source label is not one of the 2 largest predictions. Exiting")
  #  exit(-1)
  return config


def mesh_reconstruction(config=None, source_mesh=None):
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


  dnn_model = rnn_model.RnnManifoldWalkNet(params, params.n_classes, 3, model_fn,
                                   model_must_be_load=True, dump_model_visualization=False)

  if config['chose_target_label'] == True:
    config = chose_target(config, dnn_model, params, source_mesh)

  print("source label: ", config['source_label'], " target label: ", config['target_label'], " output dir: ", get_res_path(config))

  result_path, net_name = get_res_path(config)
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
  target_feature_vector = tf.one_hot(indices=config['target_label'], depth=params.n_classes)
  #return tf.one_hot(indices=labels, depth=params.n_classes)

  skip = int(params.seq_len / 2)

  loss = []

  cpos = None
  w = config['learning_weight']
  num_iter_no_change = 0
  last_dev_res = 0
  last_plt_res = 0
  skipped_iters = 0
  res_path, net_name = get_res_path(config)
  fields_needed = ['vertices', 'faces', 'edges', 'kdtree_query', 'label', 'labels', 'dataset_name', 'labels_fuzzy']
  source_pred_list = []
  target_pred_list = []
  predictions = []
  for i in range(len(config['classes_to_train'])):
    predictions.append([])

  x_axis = []
  # Defining attacks
  kl_divergence_loss = tf.keras.losses.KLDivergence()
  vertices_counter = np.ones(mesh_data['vertices'].shape)
  vertices_gradient_change_sum = np.zeros(mesh_data['vertices'].shape)
  num_times_wrong_classification = 0

  for num_iter in range(config['max_iter']):
    for i in range(config['num_walks_per_iter']):
      features, labels = dataset.mesh_data_to_walk_features(mesh_data, params)
      ftrs = tf.cast(features[:, :, :3], tf.float32)
      v_indices = features[0, :, 3].astype(np.int)
      gpu_tmpr = utils.get_gpu_temprature()

      with tf.GradientTape() as tape:
        tape.watch(ftrs)
        pred = dnn_model(ftrs, classify=True, training=False)

        # Produce the attack

        sparse_attack = 0 if config['sparse_alpha'] == 0 else w * tf.keras.losses.sparse_categorical_crossentropy(config['source_label'], pred)
        kl_div_attack = 0 if config['kl_div_alpha'] == 0 else w * kl_divergence_loss(config['source_label'], pred)
        mse_attack = 0 if config['mse_alpha'] == 0 else w * tf.keras.losses.mean_squared_error(target_feature_vector, pred[0])
        # No chamfer dist for now, as it's supposed to be between 2 sets of points, and not preds

        away_from_source_attack = config['sparse_alpha'] * sparse_attack + config['kl_div_alpha'] * kl_div_attack + config['mse_alpha'] * mse_attack

        sparse_attack = 0 if config['sparse_alpha'] == 0 else w * tf.keras.losses.sparse_categorical_crossentropy(config['target_label'], pred)
        kl_div_attack = 0 if config['kl_div_alpha'] == 0 else w * kl_divergence_loss(config['target_label'], pred)
        mse_attack = 0 if config['mse_alpha'] == 0 else w * tf.keras.losses.mean_squared_error(target_feature_vector, pred[0])
        # No chamfer dist for now, as it's supposed to be between 2 sets of points, and not preds

        #forward_target_attack = config['sparse_alpha'] * sparse_attack + config['kl_div_alpha'] * kl_div_attack + config['mse_alpha'] * mse_attack
        forward_target_attack = w * tf.keras.losses.mean_squared_error(target_feature_vector, pred[0])

        if config['attack_direction'] == 'after':
          if num_iter == config['thrs_iters_for_forward']:
            chose_target(config=config, dnn_model=dnn_model, params=params, mesh_path=mesh_data)
        attack = set_attack(config, away_from_source_attack, forward_target_attack, num_iter)


      pred = tf.reduce_sum(pred, 0)
      pred /= params.n_walks_per_model
      target_pred_brfore_attack = (pred.numpy())[config['target_label']]
      source_pred_brfore_attack = (pred.numpy())[config['source_label']]

      # If the right label is the target label and we got at least 95% pred,
      # we won't change anything and will skip the iteration.
      # That's because the part we checked already looks like we want it to.
      if (pred.numpy())[config['target_label']] > config['pred_close_enough_to_target']:
        skipped_iters += 1
        if skipped_iters < 10:
          print("skipped iter: ", num_iter)
          continue
        else:
          skipped_iters = 0

      gradients = tape.gradient(attack, ftrs)
      ftrs_after_attack_update = ftrs + gradients

      new_pred = dnn_model(ftrs_after_attack_update, classify=True, training=False)
      new_pred = tf.reduce_sum(new_pred, 0)
      new_pred /= params.n_walks_per_model


      # Check to see that we didn't update too much
      # We don't want the change to be too big, as it may result in intersections.
      # And so, we check to see if the change caused us to get closer to the target by more than 0.01.
      # If so, we will divide the change so it won't change more than 0.01
      target_pred_after_attack = (new_pred.numpy())[config['target_label']]
      source_pred_after_attack = (new_pred.numpy())[config['source_label']]

      source_pred_diff = source_pred_brfore_attack - source_pred_after_attack
      target_pred_diff = target_pred_brfore_attack - target_pred_after_attack

      #if source_pred_diff < 0 or target_pred_diff >0:
      #  print("wasn't the right direction to go")
      #  continue

      source_pred_abs_diff = abs(source_pred_brfore_attack - source_pred_after_attack)
      target_pred_abs_diff = abs(target_pred_brfore_attack - target_pred_after_attack)

      if source_pred_abs_diff > config['max_label_diff'] or target_pred_abs_diff > config['max_label_diff']:
          ratio = config['max_label_diff'] / max(source_pred_abs_diff, target_pred_abs_diff)
          gradients = gradients * ratio
          # We update the gradients accordingly
          #gradients = tape.gradient(attack, ftrs)

      # Changing the weight of the attacks if it cause too much/little effect.

      if source_pred_diff < config['no_change_pred_threshold'] or target_pred_diff < \
              config['no_change_pred_threshold']:
          num_iter_no_change += 1

      if source_pred_diff > config['no_change_pred_threshold'] or target_pred_diff > config['no_change_pred_threshold']:
        if w > 10e-3:
          w /= 2

      # Weight updates according to num of unchanged iterations
      if num_iter_no_change > config['iterations_with_no_changes']:
        if w * 2 < config['weight_threshold']:
          w *= 2
        num_iter_no_change = 0

      if num_iter % config['iter_2_change_weight'] == 0:
        if w * 10 < config['weight_threshold']:
          w *= 10

      print("iter:", num_iter, " attack:", attack.numpy(), " w:", w, " target prec:",
            (pred.numpy())[config['target_label']], " source prec:", (pred.numpy())[config['source_label']],
            " max label:", np.argmax(pred), " gpu_temp: ", gpu_tmpr)

      if np.argmax(pred) != config['source_label']:
        num_times_wrong_classification+=1
      else:
        num_times_wrong_classification = 0

      loss.append(attack.numpy())
      for i in range(len(v_indices[skip:]), len(v_indices)):
        vertices_counter[v_indices[i]] += 1
        vertices_gradient_change_sum[v_indices[i]] += gradients[0][i].numpy()
      #vertices_counter[v_indices[skip:]] += 1
      #vertices_gradient_change_sum[v_indices[skip:]] += gradients[0][skip:].numpy()

    # Updating the mesh itself
    #mesh_data['vertices'][v_indices[skip:]] += gradients[0][skip:].numpy()
    #change = np.zeros(vertices_counter.shape)
    change = vertices_gradient_change_sum/vertices_counter
    mesh_data['vertices'] += change

    # If we got the wrong classification 10 times straight
    if num_times_wrong_classification >= 10 * config['num_walks_per_iter']:
      deform_add_fields_and_dump_model(mesh_data=mesh_data, fileds_needed=fields_needed,
                                       out_fn=mesh_path[0:-4] + 'attacked.npz')

    if num_iter % 100 == 0:
      preds_to_print_str = ''
      # arange the idx of the predctions from lowest to highest
      sorted_preds = np.argsort(pred)

      #print the best 4 predictions
      for i in range(4):
        label = sorted_preds[-i -1]
        preds_to_print_str += ' pred label ' + str(label) + ': ' + str((pred.numpy())[label])

      print('\n' + str(net_name) + ' attack direction: ' + config['attack_direction'] + '\n' + preds_to_print_str +'\n')

    curr_save_image_iter = num_iter - (num_iter % config['image_save_iter'])
    if curr_save_image_iter / config['image_save_iter'] >= last_dev_res + 1 or num_iter == 0:
      print(result_path)
      cpos = dump_mesh(mesh_data, result_path, cpos, num_iter)
      last_dev_res = num_iter / config['image_save_iter']
      deform_add_fields_and_dump_model(mesh_data=mesh_data, fileds_needed=fields_needed, out_fn=result_path + '/last_model.npz')#"+ str(num_iter))

    curr_plot_iter = num_iter - (num_iter % config['plot_iter'])
    if curr_plot_iter / config['plot_iter'] >= last_plt_res + 1 or num_iter == 0:
      plot_preditions(params, dnn_model, config, mesh_data, result_path, num_iter, x_axis, source_pred_list, target_pred_list, predictions)
      last_plt_res = num_iter / config['plot_iter']


    if config['show_model_every'] > 0 and num_iter % config['show_model_every'] == 0 and num_iter > 0:
        plt.plot(loss)
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

  logdir = "../mesh_walker/runs_aug_360_must/0078-06.01.2021..15.42__camel_horse_xyz__shrec11_16-04_a/"  # '/home/alonla/mesh_walker/runs_aug_360_must/0004-11.09.2020..04.35__shrec11_16-04_A'
  dataset_path = 'datasets_processed/shrec11/16-04_a/test/*.npz'  # os.path.expanduser('~') + '/mesh_walker/datasets_processed/shrec11/16-04_a/test/*.npz'


  acc, _ = evaluate_clustering.calc_accuracy_test(logdir=logdir,
                                                        dataset_folder=dataset_path,
                                                        labels=dataset_prepare.shrec11_labels, iter2use=iter2use,
                                                        n_walks_per_model=8)
  return

def loop_over_mesh_recon():
  networks = os.listdir('../../mesh_walker/runs_aug_360_must/0000_important_runs')
  return

def main():
  np.random.seed(0)
  utils.config_gpu(1, config['gpu_to_use'])
  #check_model_accuracy()
  mesh_reconstruction(config)
  #change_one_tarj(config)

  return 0

if __name__ == '__main__':
  #for mode in ['train', 'test']:
  #  add_spheres_as_2_balls_in_dataset(dataset_path='datasets_processed/shrec11/16-04_a_changed/' + mode + '/*', mode=mode)
  main()
