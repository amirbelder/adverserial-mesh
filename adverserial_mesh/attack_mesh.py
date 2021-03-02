import argparse
import utils
import deform_mesh_new
#get hyper params from yaml
parser = argparse.ArgumentParser()
opts = parser.parse_args()
config = utils.get_config(opts.config)

if config['gpu_to_use'] >= 0:
  utils.set_single_gpu(config['gpu_to_use'])

import numpy as np
import utils
import os
import re

# meshCNN imports
#import torch
from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer



shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
  ]


def test_meshCNN():
  print('Testing MeshCNN')
  opt = TestOptions().parse()
  opt.serial_batches = True  # no shuffle
  dataset = DataLoader(opt)
  model = create_model(opt)
  writer = Writer(opt)
  # test
  #data = prepare_data()
  writer.reset_counter()
  for i, data in enumerate(dataset):
    model.set_input(data)
    ncorrect, nexamples = model.test()
    writer.update_counter(ncorrect, nexamples)
  writer.print_acc(i, writer.acc)
  return writer.acc


def eval_mesh_accuracy(mode = 'meshCNN', changed_meshes_paths = None):
  if mode is None or changed_meshes_paths is None:
    return -1
  if mode == 'meshCNN':
    for path in changed_meshes_paths:
      mesh_path = path[0:-4] + '_attacked.npz'
      orig_mesh_data = np.load(mesh_path, encoding='latin1', allow_pickle=True)
      attacked_mesh_data = {k: v for k, v in orig_mesh_data.items()}
      vertices, faces = attacked_mesh_data['vertices'], attacked_mesh_data['faces']
      v_mask = np.ones(len(vertices), dtype=bool)
      faces, face_areas = remove_non_manifolds(mesh_data, faces)
      if opt.num_aug > 1:
        faces = augmentation(mesh_data, opt, faces)
      build_gemm(mesh_data, faces, face_areas)
      if opt.num_aug > 1:
        post_augmentation(mesh_data, opt)
      mesh_data.features = extract_features(mesh_data)

    return
  return


def find_meshes_idx(config=None):
  if config is None:
    return
  files_names = os.listdir(path='datasets/shrec_16/' + shrec11_labels[config['source_label']] + '/test')
  idx = []
  for name in files_names:
    if name == 'cache':
      continue
    num, type = re.split(pattern='.', string=name)
    idx.append(num)

  meshes_paths = []
  all_files = os.listdir(path='datasets_processed/')
  for id in idx:
    mesh_path = ([file for file in all_files if file.__contains__('_'+str(id)+'_')])[0]
    meshes_paths.append('datasets_processed/' + mesh_path)
  return idx


def change_several_meshes(config=None):
  if config is None:
    return
  paths_to_source_test_meshes = find_meshes_idx(config=config)
  for path in paths_to_source_test_meshes:
    deform_mesh_new.mesh_reconstruction(config=config, target_mesh=path)
  return paths_to_source_test_meshes


def main():
  np.random.seed(0)
  utils.config_gpu(1, -1)
  test_meshCNN()
  paths_to_source_test_meshes = change_several_meshes(config=config)
  eval_mesh_accuracy(mode='meshCNN', changed_meshes_paths=paths_to_source_test_meshes)
  return 0


if __name__ == '__main__':
  main()
