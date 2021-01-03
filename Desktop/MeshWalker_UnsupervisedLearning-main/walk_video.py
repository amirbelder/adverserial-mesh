import time, glob, os, shutil, json

import trimesh
import open3d
from pylab import plt
import matplotlib
import matplotlib.animation as animation
from easydict import EasyDict
import numpy as np
from tqdm import tqdm
import pyvista as pv
import cv2

import utils
import walks
import dataset_prepare
import dataset

colors_from_gur_paper = ['#FFB8B8', '#C5C5FF', '#FFD2FF', '#FFCD88']

def model_and_walks():
  MAKE_MOVIE = 1

  dataset = 'sig17_human_seg' # shrec11 / sig17_human_seg / modelnet
  params = EasyDict()
  params.batch_size = 1
  params.seq_len = 50
  params.adjust_vertical_model = 1
  params.walk_alg = 'no_local_jumps'  # no_repeat / fastest / only_jumps / no_local_jumps
  params.net_input = ['xyz'] # edge_meshcnn , xyz , vertex_indices
  params.net_input += ['jump_indication', 'vertex_indices']
  params.n_walks_per_model = 1
  params.normalize_model = True
  params.sub_mean_for_data_augmentation = False
  params.reverse_walk = False
  params.classes_indices_to_use = None
  np.random.seed(3)
  model_fn = '/home/alonlahav/mesh_walker/datasets_processed/human_seg_from_meshcnn/train_faust__tr_reg_000_not_changed_1500.npz'
  f0 = 400

  orig_mesh_data = np.load(model_fn, encoding='latin1', allow_pickle=True)
  mesh_data = {k: v for k, v in orig_mesh_data.items()}
  mesh_extra = {'edges': mesh_data['edges']}
  vertices = mesh_data['vertices']
  mesh_extra['n_vertices'] = vertices.shape[0]
  walk, jumps = walks.get_seq_random_walk_no_jumps(mesh_extra, f0, params.seq_len)

  cpos = [[-0.36 , 0.34 , 1.14] , [-0.83 , 0.53 , -0.31] , [0.05 , 0.99 , 0.11]]
  stride = 1
  start_at = 3
  snapshots = None
  origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
  rot_stride = 1
  Rz = trimesh.transformations.rotation_matrix(rot_stride * np.pi / 180, yaxis)
  print('n jumps', np.sum(jumps))
  for n in tqdm(range(start_at, len(walk), stride)):
    #mesh_data['vertices'] = trimesh.transformations.transform_points(mesh_data['vertices'], Rz)
    last_steps = np.zeros_like(walk[:n])
    last_steps[-stride:] = 1
    if MAKE_MOVIE:
      rendered = utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], cpos=cpos, off_screen=True,
                                       opacity=0.5, show_vertices=False, walk=walk[:n], edge_colors=last_steps,
                                       dual_object=0, window_size=[int(1024), int(768)],
                                       all_colors=colors_from_gur_paper[0], line_width=1)
      if snapshots is None:
        snapshots = np.zeros((len(walk) - start_at, rendered.shape[0], rendered.shape[1], rendered.shape[2]), dtype=rendered.dtype)
      snapshots[n - start_at] = rendered
    else:
      cpos = utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], cpos=cpos, off_screen=False, line_width=2,
                                   opacity=1.0, show_vertices=False, walk=walk[:n], edge_colors=last_steps, dual_object=0)
      utils.print_cpos(cpos)

  if MAKE_MOVIE:
    fps = 10
    fig = plt.figure()
    ax1 = plt.axes()
    ax1.axis('off')
    a = snapshots[0]
    im = ax1.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)
    ttl = ax1.text(.2, 1.05, '', transform=ax1.transAxes, va='center')

    def animate_func(step):
      ttl.set_text('Step: ' + str(step))
      im.set_array(snapshots[step])
      return [im, ttl]

    anim = animation.FuncAnimation(fig, animate_func, frames=len(walk) - start_at, interval=1000 / fps)
    print('Dumping video.')
    anim.save(os.path.expanduser('~') + '/mesh_walker/walk_videos/walk.mp4', fps=fps,
              extra_args=['-vcodec', 'libx264'], dpi=800)#1200)


if __name__ == '__main__':
  model_and_walks()