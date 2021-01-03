import os

import pylab as plt
import numpy as np
import tensorflow as tf
import pyvista as pv
import trimesh


SEGMENTATION_COLORMAP = np.array(
  ((165, 242, 12), (89, 12, 89), (165, 89, 165), (242, 242, 165),
   (242, 165, 12), (89, 12, 12), (165, 12, 12), (165, 89, 242), (12, 12, 165),
   (165, 12, 89), (12, 89, 89), (165, 165, 89), (89, 242, 12), (12, 89, 165),
   (242, 242, 89), (165, 165, 165)),
  dtype=np.float32) / 255.0


def config_gpu(use_gpu=True):
  print('tf.__version__', tf.__version__)
  np.set_printoptions(suppress=True)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  try:
    if use_gpu:
      gpus = tf.config.experimental.list_physical_devices('GPU')
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    else:
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  except:
    pass


def get_numpy(x):
  if isinstance(x, np.ndarray):
    return x
  else:
    return x.numpy()


def visualize_model_walk(vertices, faces_, walks, title='', cpos=None):
  """Visualize model and walks on its"""
  faces = np.hstack([[3] + f.tolist() for f in faces_])
  surf = pv.PolyData(vertices, faces)
  p = pv.Plotter()
  p.add_mesh(surf, show_edges=True, color='white', opacity=0.6)
  p.add_mesh(pv.PolyData(surf.points), point_size=2, render_points_as_spheres=True)

  for walk in walks:
    cm = np.array(plt.get_cmap('plasma').colors)
    a = (np.arange(walk.size) * cm.shape[0] / walk.size).astype(np.int)
    colors2use = cm[a]
    all_edges = [[2, walk[i], walk[i + 1]] for i in range(len(walk) - 1)]
    walk_edges = np.hstack([edge for edge in all_edges])
    walk_mesh = pv.PolyData(vertices, walk_edges)
    p.add_mesh(walk_mesh, show_edges=True, edge_color='blue', line_width=4)
    for i, c in zip(walk, colors2use):
      if i == walk[0]:
        point_size = 20
      elif i == walk[-1]:
        point_size = 30
      else:
        point_size = 10
      p.add_mesh(pv.PolyData(surf.points[i]), color=c, point_size=point_size, render_points_as_spheres=True)
  p.camera_position = cpos
  cpos = p.show(title=title)

  return cpos


def postprocess_vertex_predictions(pred, pred_count, n_vertices, edges):
  """Average the vertex results and its neighbors to get better accuracy"""
  pred_orig = pred.copy()
  av_pred = np.zeros_like(pred_orig)
  for v in range(n_vertices):
    this_pred = pred_orig[v] / pred_count[v]
    nbrs_ids = edges[v]
    nbrs_ids = np.array([n for n in nbrs_ids if n != -1])
    if nbrs_ids.size:
      first_ring_pred = pred_orig[nbrs_ids]
      nbrs_pred = np.mean(first_ring_pred, axis=0) * 0.5
      av_pred[v] = this_pred + nbrs_pred
    else:
      av_pred[v] = this_pred
  pred = av_pred
  return pred


def posttprocess_and_dump(example, logits, walks_vertices, step=None):
  """Dump mesh results to ply file, to be visualized using Meshlab or other mesh visualizarion application."""

  # Prepare the mesh
  n_vertices = get_numpy(example['num_vertices'])[0]
  n_faces = get_numpy(example['num_triangles'])[0]
  mesh = trimesh.Trimesh(vertices=example['vertices'][0, :n_vertices, :],
                         faces=example['triangles'][0, :n_faces, :],
                         process=False)

  # Accumulate vertex predictions
  pred = np.zeros((n_vertices, logits.shape[2]))
  pred_count = 1e-6 * np.ones((n_vertices, )) # Initiated to a very small number to avoid devision by 0
  n_steps = walks_vertices.shape[1] - 1
  skip = int(n_steps / 2)
  for walk_vertices, walk_logits in zip(walks_vertices[:, 1:], logits):
    for w_step in range(skip, n_steps):
      pred[walk_vertices[w_step]] += walk_logits[w_step]
      pred_count[walk_vertices[w_step]] += 1
  #neighbors = dataset_tfg.edges2neighbors(example['edges'][0].numpy(), n_vertices, 20)
  #pred_ = postprocess_vertex_predictions(pred, pred_count, n_vertices, neighbors)

  full_accuracy = np.mean(np.argmax(pred, axis=1) == get_numpy(example['labels'][0]))

  # Dump
  colors = np.zeros((n_vertices, 3))
  for idx in range(n_vertices):
    this_pred = np.argmax(pred[idx])
    colors[idx] = SEGMENTATION_COLORMAP[this_pred]
  mesh.visual.vertex_colors = (colors * 255).astype('uint8')

  dump_path = '/tmp/mesh_walker/output_dump'
  if not os.path.isdir(dump_path):
    os.makedirs(dump_path)
  h = hash(get_numpy(example['vertices'][0]).tostring())
  out_fn = dump_path + '/predicted_' + hex(abs(h)) + '_' + str(step).zfill(8) + '.ply'
  try:
    mesh.export(out_fn)
  except:
    print('Mesh could not be dumped.')

  print(out_fn, ' was written')

  return full_accuracy



