"""
Shows a mesh, without the walking
"""

import glob, os, copy, sys

import numpy as np
import utils


def show_model(model):
    utils.visualize_model(model['vertices'], model['faces'])
    return

def load_model_from_npz(npz_path):
  if npz_path.find(':') != -1:
    npz_path = npz_path.split(':')[1]
  mesh_data = np.load(npz_path, encoding='latin1', allow_pickle=True)
  return mesh_data

if __name__ == '__main__':
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
    else:
        print("Error: Please provide .npz file path as an argument !")
        exit(0)

    model = load_model_from_npz(npz_path)
    show_model(model)