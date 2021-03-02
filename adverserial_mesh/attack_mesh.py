import argparse
import utils
import deform_mesh_new
#get hyper params from yaml
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='recon_config.yaml', help='Path to the config file.')
opts = parser.parse_args()
config = utils.get_config(opts.config)

if config['gpu_to_use'] >= 0:
  utils.set_single_gpu(config['gpu_to_use'])

import numpy as np
import utils
import os
import re


shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
  ]

def find_meshes_idx(config=None):
  if config is None:
    return
  files_names = os.listdir(path='datasets/shrec_16/' + shrec11_labels[config['source_label']] + '/test')
  idx = []
  for name in files_names:
    if name == 'cache':
      continue
    num, type = re.split(pattern='\.', string=name)
    idx.append(num)

  meshes_paths = []
  all_files = os.listdir(path='datasets_processed/shrec11/')
  for id in idx:
    mesh_path = ([file for file in all_files if file.__contains__('_'+str(id)+'_')])[0]
    meshes_paths.append('datasets_processed/shrec11/' + mesh_path)
  return meshes_paths


def change_several_meshes(config=None):
  if config is None:
    return
  paths_to_source_test_meshes = find_meshes_idx(config=config)
  for path in paths_to_source_test_meshes:
    deform_mesh_new.mesh_reconstruction(config=config, source_mesh=path)
  return paths_to_source_test_meshes


def main():
  np.random.seed(0)
  utils.config_gpu(1, -1)
  #test_meshCNN()
  paths_to_source_test_meshes = change_several_meshes(config=config)
  #eval_mesh_accuracy(mode='meshCNN', changed_meshes_paths=paths_to_source_test_meshes)
  return 0


if __name__ == '__main__':
  main()
