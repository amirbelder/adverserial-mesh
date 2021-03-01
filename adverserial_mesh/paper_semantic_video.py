import time, glob, os, shutil, json

import trimesh
import open3d
import tensorflow as tf
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
import rnn_model
from evaluate_segmentation import fill_edges as evaluate_seg_fill_edges


colors_from_gur_paper = ['#FFB8B8', '#C5C5FF', '#FFD2FF', '#FFCD88']


def dump_movie(snapshots, i2step):
  fps = 100
  fig = plt.figure()
  ax1 = plt.axes()
  ax1.axis('off')
  a = snapshots[0]
  im = ax1.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)
  ttl = ax1.text(.4, 1.05, '', transform=ax1.transAxes, va='center')

  def animate_func(i):
    ttl.set_text('Step: ' + str(i2step[i]))
    im.set_array(snapshots[i])
    return [im, ttl]

  anim = animation.FuncAnimation(fig, animate_func, frames=snapshots.shape[0], interval=1000 / fps)
  if 0:
    plt.show()
  print('Dumping video.')
  anim.save(os.path.expanduser('~') + '/mesh_walker/videos/segmentaion.mp4', fps=fps,
            extra_args=['-vcodec', 'libx264'], dpi=200)  # dpi 1200 is for good resolution


def add_axes_2_mesh(vertices, faces):
  nv = vertices.shape[0]
  vertices = np.vstack((vertices,
                        [[0, 1, 0],
                         [0, -1, 0],
                         [1, 0, 0],
                         [-1, 0, 0],
                         [0, 0, 1],
                         [0, 0, -1],
                         ]))
  faces = np.vstack((faces, [
    [nv, nv, nv + 1],
    [nv + 2, nv + 2, nv + 3],
    [nv + 4, nv + 4, nv + 5],
  ]))

  return vertices, faces


def create_edge_colors_list(faces, vertex_predictions, human_seg_cmap):
  def _calc_edge_pred(p0, p1):
    p = (p0 + p1) / 2
    if np.max(p) < 0.1:
      return None
    else:
      return np.argmax(p)
  edge_colors_list = {}
  for face in faces:
    for f in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
      edge_pred = _calc_edge_pred(vertex_predictions[f[0]], vertex_predictions[f[1]])
      if edge_pred:
        color = human_seg_cmap[edge_pred]
        if color not in edge_colors_list.keys():
          edge_colors_list[color] = []
        edge_colors_list[color].append(f)
  return edge_colors_list


def set_face_colors(mesh_faces, face_colors, pred_per_vertex):
  for face_id, face_vertices in enumerate(mesh_faces):
    face_pred = pred_per_vertex[face_vertices, :]
    mn = np.mean(face_pred, axis=0)
    if mn.sum() == 0:
      pred_id = 0
    else:
      pred_id = np.argmax(mn)
    face_colors[face_id] = pred_id


def rotate_vertices(vertices, axis, angle_deg):
  x = y = z = 0
  if axis == 0:
    x = angle_deg * np.pi / 180
  elif axis == 1:
    y = angle_deg * np.pi / 180
  elif axis == 2:
    z = angle_deg * np.pi / 180
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


def calc_segmentaion_retults(rel_run_dir, model_fn, pathname_expansion, output_fn):
  #logdir = os.path.expanduser('~') + '/mesh_walker/runs_test/' + rel_run_dir + '/'
  logdir = '/media/alonlahav/0.5T/runs/runs_test/' + rel_run_dir + '/'

  with open(logdir + '/params.txt') as fp:
    params = EasyDict(json.load(fp))
  params.walk_alg = 'random_global_jumps' # !!!!!!!!!!!!!! #
  params.batch_size = 1
  params.n_walks_per_model = 1
  params.net_input.append('vertex_indices')
  params.seq_len = 10000

  test_dataset = dataset.tf_mesh_dataset(params, pathname_expansion, mode=params.network_task)
  shape_model_fn = pathname_expansion
  mesh_data = np.load(shape_model_fn, encoding='latin1', allow_pickle=True)

  dummy_optimizer = tf.keras.optimizers.Adam(lr=0.1, clipnorm=params.gradient_clip_th)
  dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim - 1, logdir + '/' + model_fn,
                                     model_must_be_load=True, dump_model_visualization=False, optimizer=dummy_optimizer)

  ftrs, labels = dataset.mesh_data_to_walk_features(mesh_data, dataset.dataset_params_list[0])
  model_ftrs = ftrs[:, :, :-1]
  all_seq = ftrs[:, :, -1]

  predictions = dnn_model(model_ftrs, training=False).numpy()
  all_seq = all_seq.reshape(-1).astype(np.int32)
  predictions4vertex = predictions.reshape((-1, predictions.shape[-1]))
  pred_label_per_vertex = np.argmax(predictions4vertex, axis=1)
  labels = labels.flatten()
  this_acc = np.mean(labels[1:] == pred_label_per_vertex)
  print('Acc:', this_acc)

  m = {'vertices': mesh_data['vertices'],
       'faces': mesh_data['faces'],
       'walk': all_seq,
       'shape_model_fn': shape_model_fn,
       'gt_labels': mesh_data['labels'],
       'predictions4vertex': predictions4vertex,
       }
  folder = os.path.split(output_fn)[0]
  if not os.path.isdir(folder):
    os.makedirs(folder)
    os.makedirs(folder + '/output_images')
  np.savez(output_fn, **m)

  return m


def gt_edges_2_colors_list(gt_edges2color, faces):
  model = {'faces': faces}
  evaluate_seg_fill_edges(model)
  edges_meshcnn = model['edges_meshcnn']
  gt_edge_colors_list = []
  for c in range(1, 9):
    i = np.where(gt_edges2color == c)[0]
    this_edges = edges_meshcnn[i]
    gt_edge_colors_list.append(this_edges)
  return gt_edge_colors_list


def get_cpos(dataset, shape_id):
  if dataset == 'human_seg':
    if shape_id == 13:
      return [[0.45, 0.33, -4.79], [0.08, -0.09, 0.0], [-0.03, 1.0, 0.08]]
    elif shape_id == 2:
      return [[3.57 , 0.93 , 3.16] , [0.08 , -0.09 , 0.0] , [-0.19 , 0.98 , -0.1]]
    else:
      return [[-4.52 , -0.0 , -1.44] , [0.08 , -0.09 , -0.0] , [0.01 , 1.0 , 0.03]]
  if dataset == 'coseg-vases':
    return [[-4.05, 1.95, -1.41], [0.08, -0.09, 0.0], [0.4, 0.91, 0.15]]
  if dataset == 'coseg-aliens':
    return [[2.85 , 0.38 , 3.91] , [0.08 , -0.09 , 0.0] , [-0.03 , 0.99 , -0.1]]
  if dataset == 'coseg-chairs':
    return [[2.66 , 1.35 , 3.8] , [0.08 , -0.09 , 0.0] , [-0.14 , 0.95 , -0.27]]


def make_segmentaion_figures(dataset, shape_id=1):
  MAKE_MOVIE = 1
  COLORIZE_EDGES = 1

  human_seg_cmap     = ['black', 'red',  'green', 'blue',      'orange',    'magenta', 'yellow',    'cyan',     'lightgreen']
  sigg17_part_labels = ['---',   'head', 'hand',  'lower-arm', 'upper-arm', 'body',   'upper-lag', 'lower-leg', 'foot']

  path_output = os.path.expanduser('~') + '/runs/images/'

  # Calc my results
  my_res_fn = path_output + dataset + '/my_segmentation_res_' + str(shape_id) + '.npz'
  if 1:
    pathname_expansion = os.path.expanduser('~') + '/mesh_walker/datasets_processed/' + 'sig17_seg_benchmark-4k/test__shrec_' + \
                         str(shape_id) + '_simplified_to_4000.npz'
    rel_run_dir = '/0017-29.06.2020..08.15__human_seg-4k/'
    model_fn = 'learned_model2keep__00140008.keras'
    mesh_data = calc_segmentaion_retults(rel_run_dir, model_fn, pathname_expansion, output_fn=my_res_fn)
  else:
    mesh_data = np.load(my_res_fn, encoding='latin1', allow_pickle=True)

  # Show my results
  cpos = get_cpos(dataset, shape_id)

  predictions = mesh_data['predictions4vertex']
  stride = 1
  if MAKE_MOVIE:
    start = 0
    if 0:
      end = 9900
      frames2skip = [[500, 1000], [1300, 2000], [2300, 3000], [3300, 4000], [4300, 5000], [5300, 6000], [6300, 6900],
                     [7300, 7900], [8300, 8900], [9300, 9800]]
    elif 1:
      end = 5200
      frames2skip = [[300, 1000], [1100, 2000], [2100, 3000], [3300, 4000], [4300, 5000]]
    else:
      start = 1000
      end = 1200
      frames2skip = []
  else:
    start = 100
    end = predictions.shape[0] - 10

  n_frame2skip = int(np.sum([x[1] - x[0] - 1 for x in frames2skip]))
  n_total_frames = end - start - n_frame2skip

  snapshots = None
  pred_per_vertex = np.zeros((mesh_data['vertices'].shape[0], predictions.shape[1]))
  next_frame = 0
  for this_step in range(0, start):
    pred_per_vertex[mesh_data['walk'][this_step]] += predictions[this_step]

  vertices = mesh_data['vertices']

  i2step = []
  rot_stride = 360 / n_total_frames
  for this_step in tqdm(range(start, end)):
    pred_per_vertex[mesh_data['walk'][this_step]] += predictions[this_step]
    skip_this_step = np.any([this_step > x[0] and this_step < x[1] for x in frames2skip])
    if skip_this_step:
      continue
    rotate_vertices(vertices, 1, rot_stride)

    walk2show = mesh_data['walk'][this_step - 3:this_step + 2]
    last_steps = np.zeros_like(walk2show)
    if last_steps.size:
      last_steps[-1] = 1

    if COLORIZE_EDGES:
      face_colors = None
      show_edges = True
      line_width = 0.5
      window_size = [768, 512]
      edge_color_a = 'gray'
      edge_colors_list = create_edge_colors_list(mesh_data['faces'], pred_per_vertex, human_seg_cmap)
      dual_object = None
    else:
      face_colors = np.zeros((mesh_data['faces'].shape[0],))
      set_face_colors(mesh_data['faces'], face_colors, pred_per_vertex)
      show_edges = False
      line_width = 5
      window_size = [384, 256]
      edge_color_a = 'whhite'
      edge_colors_list = None
      dual_object = 2

    vis_res = utils.visualize_model(vertices, mesh_data['faces'], cpos=cpos, off_screen=MAKE_MOVIE, window_size=window_size,
                                    face_colors=face_colors, edge_color_a=edge_color_a, edge_colors_list=edge_colors_list,
                                    dual_object=dual_object, show_vertices=False, opacity=1.0,
                                    cmap=human_seg_cmap, walk=walk2show, edge_colors=last_steps, show_edges=show_edges, line_width=line_width)
    if MAKE_MOVIE:
      rendered = vis_res
      if snapshots is None:
        snapshots = np.zeros((n_total_frames,
                              rendered.shape[0], rendered.shape[1], rendered.shape[2]),
                              dtype=rendered.dtype)
      snapshots[next_frame] = rendered
      i2step.append(this_step)
      next_frame += 1
    else:
      utils.print_cpos(vis_res)

  if MAKE_MOVIE:
    dump_movie(snapshots, i2step)

'''
To convert to gif:
cd /home/alonlahav/mesh_walker/videos/
ffmpeg -i segmentaion-full.mp4 segmentaion.gif
'''

if __name__ == '__main__':
  utils.config_gpu(1)
  np.random.seed(0)
  tf.random.set_seed(0)

  make_segmentaion_figures('human_seg', 2)
