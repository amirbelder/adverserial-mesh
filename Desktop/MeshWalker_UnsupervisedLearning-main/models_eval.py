import os, shutil, time, copy, glob
from easydict import EasyDict
import json
import platform

import cv2
import numpy as np
import tensorflow as tf
import trimesh, open3d
import pyvista as pv
import csv
import pandas as pd

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
  folder_name = config['trained_model'].split('/')[-1]
  if len(folder_name) > 0:
    res_path = '../../mesh_walker/' +folder_name +'/' + shrec11_labels[config['source_label']] + '_to_' + shrec11_labels[config['target_label']]
  else:
    res_path = '../../mesh_walker/'  + shrec11_labels[config['source_label']] + '_to_' + shrec11_labels[config['target_label']]
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

def eval_npz_preds(config):
  walk_len = 2000
  #dog2
  npz_paths_dog2 = ['datasets_processed/shrec11/16-04_a/train/T504_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T93_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T207_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T507_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T8_not_changed_500.npz']
  #dog1
  npz_paths_dog1 = ['datasets_processed/shrec11/16-04_a/train/T476_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T144_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T331_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T125_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T393_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T436_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T197_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T367_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T409_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T373_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T193_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T309_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T99_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T136_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T354_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T203_not_changed_500.npz']
  #horse
  npz_paths_horse = ['datasets_processed/shrec11/16-04_a/train/T295_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T348_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T364_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T184_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T301_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T304_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T454_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T343_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T142_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T303_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T117_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T485_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T234_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T579_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T402_not_changed_500.npz',
                     'datasets_processed/shrec11/16-04_a/train/T529_not_changed_500.npz']
  #camel
  npz_paths_camel = ['datasets_processed/shrec11/16-04_a/train/T187_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T219_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T336_not_changed_500.npz'	,'datasets_processed/shrec11/16-04_a/train/T439_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T481_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T300_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T464_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T497_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T536_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T52_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T431_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T92_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T35_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T538_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T453_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T398_not_changed_500.npz']

  res_file = open("dog_1_dog_2_horse_camel_eval_npz.txt", "a")
  paths = []
  orig_labels = []
  dog1_preds = []
  dog2_preds = []
  horse_preds = []
  camel_preds = []

  for npz_paths in [npz_paths_dog2, npz_paths_horse, npz_paths_camel, npz_paths_dog1]:
      with open(config['trained_model'] + '/params.txt') as fp:
          params = EasyDict(json.load(fp))
      model_fn = tf.train.latest_checkpoint(config['trained_model'])
      params.batch_size = 1
      params.seq_len = walk_len
      params.n_walks_per_model = 16
      params.set_seq_len_by_n_faces = False
      params.data_augmentaion_vertices_functions = []
      params.label_per_step = False
      params.n_target_vrt_to_norm_walk = 0
      params.net_input += ['vertex_indices']
      dataset.setup_features_params(params, params)
      dataset.mesh_data_to_walk_features.SET_SEED_WALK = False

      dnn_model = rnn_model.RnnManifoldWalkNet(params, params.n_classes, 3, model_fn,
                                       model_must_be_load=True, dump_model_visualization=False)


      shrec11_labels = [
      'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
      'ants',       'rabbit',   'dog1',       'snake',      'bird2',
      'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
      'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
      'alien',      'octopus',  'cat',        'woman',      'spiders',
      'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
      ]


      for path in npz_paths:
        orig_mesh_data = np.load(path, encoding='latin1', allow_pickle=True)
        mesh_data = {k: v for k, v in orig_mesh_data.items()}

        features, labels = dataset.mesh_data_to_walk_features(mesh_data, params)
        ftrs = tf.cast(features[:, :, :3], tf.float32)
        preds = dnn_model(ftrs, classify=True, training=False)
        sum_pred = tf.reduce_sum(preds, 0)
        dog1_pred = (sum_pred.numpy())[7] / 16
        dog2_pred = (sum_pred.numpy())[4] / 16
        horse_pred = (sum_pred.numpy())[15] / 16
        camel_pred = (sum_pred.numpy())[25] / 16
        paths.append(path)
        orig_labels.append(mesh_data['label'])
        dog1_preds.append(dog1_pred)
        dog2_preds.append(dog2_pred)
        horse_preds.append(horse_pred)
        camel_preds.append(camel_pred)

  info_dict = {'Path': paths, 'Orig Labels': orig_labels, 'dog1_pred': dog1_preds, 'dog2_preds': dog2_preds, 'horse_preds': horse_preds, 'camel_preds': camel_preds}
  data = pd.DataFrame(info_dict)
  data.to_csv(path_or_buf="dog_1_dog_2_horse_camel_eval_npz.csv")


def mesh_reconstruction(config):

  nets = os.listdir('../../mesh_walker/runs_aug_360_must/0000_important_runs')
  walk_lens = [200, 500, 1000, 2000]
  #dog2
  npz_paths_2 = ['datasets_processed/shrec11/16-04_a/train/T504_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T93_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T207_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T507_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T8_not_changed_500.npz']
  #dog1
  npz_paths_1 = ['datasets_processed/shrec11/16-04_a/train/T476_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T144_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T331_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T125_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T393_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T436_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T197_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T367_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T409_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T373_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T193_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T309_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T99_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T136_not_changed_500.npz', 'datasets_processed/shrec11/16-04_a/train/T354_not_changed_500.npz',	'datasets_processed/shrec11/16-04_a/train/T203_not_changed_500.npz']

  res_file = open("dog_1_dog_2_eval_nets.txt", "a")

  for net in nets:
      for walk_len in walk_lens:
          for npz_paths in [npz_paths_1, npz_paths_2]:
                  with open('../../mesh_walker/runs_aug_360_must/0000_important_runs' + '/' +  net + '/params.txt') as fp:
                    params = EasyDict(json.load(fp))
                  model_fn = tf.train.latest_checkpoint('../../mesh_walker/runs_aug_360_must/0000_important_runs' + '/' + net)
                  params.batch_size = 1
                  params.seq_len = walk_len
                  params.n_walks_per_model = 8
                  params.set_seq_len_by_n_faces = False
                  params.data_augmentaion_vertices_functions = []
                  params.label_per_step = False
                  params.n_target_vrt_to_norm_walk = 0
                  params.net_input += ['vertex_indices']
                  dataset.setup_features_params(params, params)
                  dataset.mesh_data_to_walk_features.SET_SEED_WALK = False


                  dnn_model = rnn_model.RnnManifoldWalkNet(params, params.n_classes, 3, model_fn,
                                                   model_must_be_load=True, dump_model_visualization=False)

                  avg_dog2 = 0
                  avg_dog1 = 0
                  #npz_paths = npz_paths_1 + npz_paths_2
                  shrec11_labels = [
                  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
                  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
                  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
                  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
                  'alien',      'octopus',  'cat',        'woman',      'spiders',
                  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
                  ]

                  for path in npz_paths:
                    orig_mesh_data = np.load(path, encoding='latin1', allow_pickle=True)
                    mesh_data = {k: v for k, v in orig_mesh_data.items()}
                    params.n_walks_per_model = 8

                    features, labels = dataset.mesh_data_to_walk_features(mesh_data, params)
                    ftrs = tf.cast(features[:, :, :3], tf.float32)
                    eight_pred = dnn_model(ftrs, classify=True, training=False)
                    sum_pred = tf.reduce_sum(eight_pred, 0)
                    avg_dog1 += (sum_pred.numpy())[7] / 8
                    avg_dog2 += (sum_pred.numpy())[4] / 8

                  avg_dog1 = avg_dog1/len(npz_paths)
                  avg_dog2 = avg_dog2/len(npz_paths)
                  line = "net: " + str(net) + " walk len: " + str(walk_len) + " avg_dog1: " + str(avg_dog1) + " avg_dog2: " + str(avg_dog2) + " len npz: " + str(len(npz_paths)) + '\n'
                  res_file.writelines([line])
                  print("net: ", net, " walk len: ", walk_len, " avg_dog1: ", avg_dog1, " avg_dog2: ", avg_dog2, " len npz: ", len(npz_paths))
  res_file.close()
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
  #mesh_reconstruction(config)
  eval_npz_preds(config)

  return 0

if __name__ == '__main__':
  main()
