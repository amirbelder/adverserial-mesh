import os, shutil, psutil, json, copy
import time, datetime
import itertools
from easydict import EasyDict

import pylab as plt
import numpy as np
import tensorflow as tf
import pyvista as pv
import trimesh

#import dnn_cad_seq
import evaluate_classification
import evaluate_segmentation

SEGMENTATION_COLORMAP = np.array(
  ((165, 242, 12), (89, 12, 89), (165, 89, 165), (242, 242, 165),
   (242, 165, 12), (89, 12, 12), (165, 12, 12), (165, 89, 242), (12, 12, 165),
   (165, 12, 89), (12, 89, 89), (165, 165, 89), (89, 242, 12), (12, 89, 165),
   (242, 242, 89), (165, 165, 165)),
  dtype=np.float32) / 255.0


class color:
  PURPLE = '\033[95m'
  CYAN = '\033[96m'
  DARKCYAN = '\033[36m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  RED = '\033[91m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'
  END = '\033[0m'

colors_list = ['black',       'red',    'green',        'blue',       'orange',     'magenta',    'yellow',   'cyan',
               'gray',        'brown',  'lightgreen',   'steelblue',  'lightcoral', 'pink',       'gold',
                                        'olive',        'darkblue',   'salmon',     'deeppink',
                                        'lime',                       'tomato',
                                                                      'sienna',
               ]

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


def backup_python_files_and_params(params):
  save_id = 0
  while 1:
    code_log_folder = params.logdir + '/.' + str(save_id)
    if not os.path.isdir(code_log_folder):
      os.makedirs(code_log_folder)
      for file in os.listdir():
        if file.endswith('py'):
          shutil.copyfile(file, code_log_folder + '/' + file)
      break
    else:
      save_id += 1

  # Dump params to text file
  try:
    prm2dump = copy.deepcopy(params)
    if 'hyper_params' in prm2dump.keys():
      prm2dump.hyper_params = str(prm2dump.hyper_params)
      prm2dump.hparams_metrics = prm2dump.hparams_metrics[0]._display_name
      for l in prm2dump.net:
        l['layer_function'] = 'layer_function'
    with open(params.logdir + '/params.txt', 'w') as fp:
      json.dump(prm2dump, fp, indent=2, sort_keys=True)
  except:
    pass


def get_run_folder(root_dir, str2add='', cont_run_number=False):
  try:
    all_runs = os.listdir(root_dir)
    run_ids = [int(d.split('-')[0]) for d in all_runs if '-' in d]
    if cont_run_number:
      n = [i for i, m in enumerate(run_ids) if m == cont_run_number][0]
      run_dir = root_dir + all_runs[n]
      print('Continue to run at:', run_dir)
      return run_dir
    n = np.sort(run_ids)[-1]
  except:
    n = 0
  now = datetime.datetime.now()
  return root_dir + str(n + 1).zfill(4) + '-' + now.strftime("%d.%m.%Y..%H.%M") + str2add


def index2color(idx):
  return SEGMENTATION_COLORMAP[np.array(idx).astype(np.int)]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          show_txt=True):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  if show_txt:
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()


def update_lerning_rate_in_optimizer(n_times_loss_stable, method, optimizer, params):
  iter = optimizer.iterations.numpy()
  if method == 'triangle' and n_times_loss_stable >= 1:
    iter = iter % params.cyclic_lr_period
    far_from_mid = np.abs(iter - params.cyclic_lr_period / 2)
    fraction_from_mid = np.abs(params.cyclic_lr_period / 2 - far_from_mid) / (params.cyclic_lr_period / 2)
    factor = fraction_from_mid + (1 - fraction_from_mid) * params.min_lr_factor
    optimizer.learning_rate.assign(params.learning_rate * factor)

  if method == 'steps':
    for i in range(len(params.learning_rate_steps) - 1):
      if iter >= params.learning_rate_steps[i] and iter < params.learning_rate_steps[i + 1]:
        lr = params.learning_rate[i]
        optimizer.learning_rate.assign(lr)

last_free_mem = np.inf
def check_mem_and_exit_if_full():
  global last_free_mem
  free_mem = psutil.virtual_memory().available + psutil.swap_memory().free
  free_mem_gb = round(free_mem / 1024 / 1024 / 1024, 2)
  if last_free_mem > free_mem_gb + 0.25:
    last_free_mem = free_mem_gb
    print('free_mem', free_mem_gb, 'GB')
  if free_mem_gb < 1:
    print('!!! Exiting due to memory full !!!')
    exit(111)
  return free_mem_gb


def visualize_model_loop_on_each_color(vertices, faces, title=' ', vertex_colors_idx=None, cpos=None):
  clrs = np.unique(vertex_colors_idx)
  for c in clrs:
    if c == -1:
      continue
    v_colors = -1 * np.ones((vertices.shape[0])).astype(np.int)
    i = np.where(vertex_colors_idx==c)
    v_colors[i] = c
    print(c, i[0].size)
    visualize_model(vertices, faces, title=' ', vertex_colors_idx=v_colors, cpos=cpos)


def visualize_model_walk(vertices, faces_, walk, jumps, title='', cpos=None):
  faces = np.hstack([[3] + f.tolist() for f in faces_])
  surf = pv.PolyData(vertices, faces)
  p = pv.Plotter()
  p.add_mesh(surf, show_edges=True, color='white', opacity=0.6)
  p.add_mesh(pv.PolyData(surf.points), point_size=2, render_points_as_spheres=True)

  cm = np.array(plt.get_cmap('plasma').colors)
  a = (np.arange(walk.size) * cm.shape[0] / walk.size).astype(np.int)
  colors2use = cm[a]
  all_edges = [[2, walk[i], walk[i + 1]] for i in range(len(walk) - 1)]
  if np.any(1 - jumps):
    walk_edges = np.hstack([edge for edge, jump in zip(all_edges, jumps[1:]) if not jump])
    walk_mesh = pv.PolyData(vertices, walk_edges)
    p.add_mesh(walk_mesh, show_edges=True, edge_color='blue', line_width=4)
  if np.any(jumps[1:]):
    jump_edges = np.hstack([edge for edge, jump in zip(all_edges, jumps[1:]) if jump])
    walk_mesh = pv.PolyData(vertices, jump_edges)
    p.add_mesh(walk_mesh, show_edges=True, edge_color='red', line_width=4)
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


def merge_some_models_for_visualization(models):
  def _get_faces(mesh):
    if type(mesh) is dict:
      return mesh['faces']
    else:
      return np.asarray(mesh.triangles)
  def _get_vertices(mesh):
    if type(mesh) is dict:
      return mesh['faces']
    else:
      return np.asarray(mesh.vertices)
  all_faces = _get_faces(models[0])
  all_vertices = _get_vertices(models[0])
  x_shift = all_vertices.max(axis=0)[0]
  for model in models[1:]:
    this_faces = _get_faces(model) + all_vertices.shape[0]
    all_faces = np.vstack((all_faces, this_faces))
    this_vertices = _get_vertices(model).copy()
    this_vertices[:, 0] += x_shift - this_vertices[:, 0].min() + (this_vertices[:, 0].max() - this_vertices[:, 0].min()) * 0.1
    all_vertices = np.vstack((all_vertices, this_vertices))
    x_shift = all_vertices.max(axis=0)[0]
  return all_vertices, all_faces


def visualize_model(vertices_, faces_, title=' ', vertex_colors_idx=None, cpos=None, v_size=None, off_screen=False, walk=None, opacity=0.7,
                    all_colors='white', face_colors=None, show_vertices=False, cmap=None, edge_colors=None, edge_color_a='white', dual_object=None,
                    dual_object_shift=0.7,window_size=[1024, 768], line_width=0.2, show_edges=False, edge_colors_list=None, point_size=15, show_surface=True,
                    save_fn=None):
  p = pv.Plotter(off_screen=off_screen, window_size=(int(window_size[0]), int(window_size[1])), title=title)
  if dual_object is None:
    n_obj2show = 1
  else:
    n_obj2show = 2
  for pos in range(n_obj2show):
    faces = np.hstack([[3] + f.tolist() for f in faces_])
    vertices = vertices_.copy()
    if dual_object is not None:
      v_shift = [0, 0, 0]
      if pos == 0:
        v_shift[dual_object] -= dual_object_shift
      else:
        vertices[:, 2 - dual_object] *= -1
        vertices[:, dual_object] *= -1
        v_shift[dual_object] += dual_object_shift
      vertices += v_shift
    surf = pv.PolyData(vertices, faces)
    if show_surface:
      p.add_mesh(surf, show_edges=show_edges, edge_color=edge_color_a, color=all_colors, opacity=opacity, smooth_shading=True,
                 scalars=face_colors, cmap=cmap, line_width=line_width)
    if show_vertices is not None:
      p.add_mesh(pv.PolyData(vertices[show_vertices]), point_size=point_size, render_points_as_spheres=True, color='red')
    if show_vertices is not None and vertex_colors_idx is not None:
      if type(cmap) is list:
        colors = cmap
      else:
        colors = 'ygbkmcywr'
      for c in np.unique(vertex_colors_idx):
        if c != -1:
          idxs = np.where(vertex_colors_idx == c)[0]
          if v_size is None:
            p.add_mesh(pv.PolyData(surf.points[idxs]), color=colors[c], point_size=point_size, render_points_as_spheres=True)
          else:
            for i in idxs:
              p.add_mesh(pv.PolyData(surf.points[i]), color=colors[c % len(colors)], point_size=v_size[i],
                         render_points_as_spheres=True)
    colors = ['blue', 'red', 'yellow', 'lime', 'orange', 'black', 'pink', 'lightblue', 'lightgreen']
    if type(walk) is list:
      for i, w in enumerate(walk):
        all_edges = [[2, w[i], w[i + 1]] for i in range(len(w) - 1)]
        walk_edges = np.hstack([edge for edge in all_edges])
        walk_mesh = pv.PolyData(vertices, walk_edges)
        p.add_mesh(walk_mesh, show_edges=True, line_width=line_width * 4, edge_color=colors[i])
    elif walk is not None and walk.size > 1:
      all_edges = [[2, walk[i], walk[i + 1]] for i in range(len(walk) - 1)]
      if edge_colors is None:
        edge_colors = np.zeros_like(walk)
      elif edge_colors == 'use_cmap':
        if 0:
          walk_edges = np.hstack([edge for edge in all_edges])
          walk_mesh = pv.PolyData(vertices, walk_edges)
          scalars = (np.arange(len(all_edges)) / len(all_edges) * 255).astype(np.int)
          p.add_mesh(walk_mesh, show_edges=True, line_width=line_width, cmap='Blues', scalars=scalars)
        else:
          for i, edge in enumerate(all_edges):
            walk_edges = np.array(edge)
            walk_mesh = pv.PolyData(vertices, walk_edges)
            edge_color = (1.0 - i / len(all_edges), 0.0, i / len(all_edges))
            p.add_mesh(walk_mesh, show_edges=True, line_width=line_width, edge_color=edge_color)
      elif type(edge_colors) is str:
        walk_edges = np.hstack([edge for edge in all_edges])
        walk_mesh = pv.PolyData(vertices, walk_edges)
        p.add_mesh(walk_mesh, show_edges=True, line_width=line_width * 4, edge_color=edge_colors)
      else:
        for this_e_color in range(np.max(edge_colors) + 1):
          this_edges = (edge_colors == this_e_color)
          if np.any(this_edges):
            walk_edges = np.hstack([edge for edge, clr in zip(all_edges, this_edges[1:]) if clr])
            walk_mesh = pv.PolyData(vertices, walk_edges)
            p.add_mesh(walk_mesh, show_edges=True, edge_color=colors[this_e_color], line_width=line_width)
  if edge_colors_list is not None:
    t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces_, process=False)
    for clr, edges in edge_colors_list.items():
      if clr.find(':') == -1:
        vertices2use = vertices
        this_edges_ = [[2, e[0], e[1]] for e in edges]
        this_edges = np.hstack([edge for edge in this_edges_])
        walk_mesh = pv.PolyData(vertices2use, this_edges)
        p.add_mesh(walk_mesh, show_edges=True, edge_color=clr, line_width=line_width)
      else:
        clr_1st, clr_2nd = clr.split(':')
        vertices2use = []
        this_edges_1st_clr = []
        this_edges_2nd_clr = []
        for e in edges:
          mean_normal = (t_mesh.vertex_normals[e[0]] + t_mesh.vertex_normals[e[1]]) / 2
          v0 = vertices[e[0]]
          v1 = vertices[e[1]]
          v_ = (v0 + v1) / 2 + mean_normal * np.linalg.norm(v0 - v1) * 0.001
          vertices2use.append(v0)
          vertices2use.append(v1)
          vertices2use.append(v_)
          this_edges_1st_clr.append([2, len(vertices2use) - 3, len(vertices2use) - 1])
          this_edges_2nd_clr.append([2, len(vertices2use) - 2, len(vertices2use) - 1])
        vertices2use = np.array(vertices2use)
        this_edges = np.hstack([edge for edge in this_edges_1st_clr])
        walk_mesh = pv.PolyData(vertices2use, this_edges)
        p.add_mesh(walk_mesh, show_edges=True, edge_color=clr_1st, line_width=line_width)
        this_edges = np.hstack([edge for edge in this_edges_2nd_clr])
        walk_mesh = pv.PolyData(vertices2use, this_edges)
        p.add_mesh(walk_mesh, show_edges=True, edge_color=clr_2nd, line_width=line_width)

  focal_point = (0., 0., 0.)
  p.camera_position = [(-3, -3, 2), focal_point, (0., 0., 1.)]
  #p.show_bounds(grid='front', location='outer', all_edges=True)
  #min_v = np.min(vertices, axis=0)
  #p.add_mesh(pv.Plane(center=(0, -min_v[1], 0), direction=(0, 1, 0), i_size=3, j_size=3))
  p.set_background("#AAAAAA", top="White")
  if off_screen:
    if not os.path.exists(os.path.dirname(save_fn)):
      os.makedirs(os.path.dirname(save_fn))
    rendered = p.screenshot(window_size=(2048, 768*2))   #filename=save_fn + '.png'
    fig = plt.figure(frameon=False)
    # fig.set_size_inches(window_size[1]/ 100, window_size[0] / 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(rendered)
    ax.text(400, 80, title, fontsize=14)
    fig.savefig(save_fn + '.png', dpi=800)
    p.close()
    plt.close('all')
    return rendered
  else:
    cpos = p.show(title=title)

  return cpos


def visualize_npz(npz_path):
  import dataset
  mesh = dataset.load_model_from_npz(npz_path)
  visualize_model(mesh['vertices'], mesh['faces'])


def add_walk(p, vertices, walk, colors, line_width=1.0):
  all_edges = [[2, walk[i], walk[i + 1]] for i in range(len(walk) - 1)]
  walk_edges = np.hstack([edge for edge in all_edges])
  walk_mesh = pv.PolyData(vertices, walk_edges)
  p.add_mesh(walk_mesh, show_edges=True, line_width=line_width * 4, edge_color=colors[i])


def print_cpos(cpos):
  s = '['
  for i, c in enumerate(cpos):
    s += '['
    for j, n in enumerate(c):
      s += str(round(n, 2))
      if j != 2:
        s += ' , '
    s += ']'
    if i != 2:
      s += ' , '
  s += ']'
  print(s)


next_iter_to_keep = 0 # Should be set by -train_val- function, each time job starts
def save_model_if_needed(iterations, dnn_model, params, override=False):
  global next_iter_to_keep
  iter_th = 20000
  keep = iterations.numpy() >= next_iter_to_keep
  dnn_model.save_weights(params.logdir, iterations.numpy(), keep=keep)

  if keep or override:
    if iterations < iter_th:
      next_iter_to_keep = iterations * 2
    else:
      next_iter_to_keep = int(iterations / iter_th) * iter_th + iter_th
    if params.full_accuracy_test is not None:
      if params.network_task == 'semantic_segmentation':
        if 'scannet' in params.logdir:
          full_coverage = next_iter_to_keep > 200000   # TODO: update. full coverage is a VERY long run. need to see when to do (end of training?)
          accuracy, _ = evaluate_segmentation.calc_accuracy_test_scannet(params=params, dnn_model=dnn_model, verbose_level=0,
                                                                 full_coverage= full_coverage,**params.full_accuracy_test)
        else:
          accuracy, _ = evaluate_segmentation.calc_accuracy_test(params=params, dnn_model=dnn_model, verbose_level=0,
                                                                 **params.full_accuracy_test)
      elif params.network_task == 'classification':
        accuracy, _ = evaluate_classification.calc_accuracy_test(params=params, dnn_model=dnn_model, verbose_level=0, **params.full_accuracy_test)
      for k, v in accuracy.items():
        with open(params.logdir + '/log.txt', 'at') as f:
          f.write(k + ': ' + '{:2.3f}'.format(v) + '%, Iter: {}'.format(iterations.numpy()) + '\n')
        tf.summary.scalar('full_evaluation_test/{}'.format(k), v, step=iterations)


def get_dataset_type_from_name(tf_names):
  name_str = tf_names[0].numpy().decode()
  return name_str[:name_str.find(':')]


def get_model_name_from_npz_fn(npz_fn):
  if 'FUTURE' in npz_fn:
    fn = npz_fn.split('/')[-2]
    n_faces = int(npz_fn.split('_')[-1].split('.')[0])
  else:
    fn = npz_fn.split('/')[-1].split('.')[-2]
    sp_fn = fn.split('_')
    if npz_fn.find('/shrec11') == -1:
      sp_fn = sp_fn[1:]
    i = np.where([s.isdigit() for s in sp_fn])[0][0]
    model_name = '_'.join(sp_fn[:i + 1])
    n_faces = int(sp_fn[-1])

  return model_name, n_faces


def colorize_and_dump_model(model, idxs2color, out_fn, clrs=None, norm_clrs=True, show=False, vertex_colors=None,
                            idxs2color_faces=True, verbose=True):
  mesh = trimesh.Trimesh(vertices=model.vertices, faces=model.faces, process=False)

  some_colors = False
  if len(idxs2color) and type(idxs2color[0]) is list:
    some_colors = True
  faces_colored = 0
  if idxs2color_faces:
    face_colors = np.zeros((model.faces.shape[0], 3))
  else:
    face_colors = np.zeros((model.vertices.shape[0], 3))
  alpha = np.zeros((model.faces.shape[0], 1))
  f = []
  for idx in range(model.faces.shape[0]):
    r = 0
    g = 0
    b = 0
    if clrs is not None:
      (r, g, b) = clrs[idx]
    elif some_colors:
      for c_idx_, l in enumerate(idxs2color):
        if idx in l:
          c_idx = c_idx_ % len(idx2color)
          r, g, b = idx2color[c_idx]
          faces_colored += 1
          break
    else:
      if idx in idxs2color:
        b = int(255 * np.where(np.array(idxs2color) == idx)[0][0] / len(idxs2color))
        r = 100
    if r or b or g:
      face_colors[idx] = [r, g, b]
      alpha[idx] = 255

  if vertex_colors is not None:
    vertex_colors *= 255
    mesh.visual.vertex_colors = vertex_colors
  elif not idxs2color_faces:
    mesh.visual.vertex_colors = face_colors.astype('uint8')
  else:
    if norm_clrs:
      face_colors -= face_colors.min()
      face_colors /= face_colors.max() / 255
    else:
      face_colors *= 255

    face_colors = np.hstack((face_colors, alpha))

    mesh.visual.face_colors = face_colors.astype('uint8')

  trimesh.repair.fix_normals(mesh)
  try:
    mesh.export(out_fn)
  except:
    print('Mesh could not be dumped.')

  if show:
    mesh.show()

  if verbose:
    print(out_fn, ' was written')




if __name__ == '__main__':
  # TODO: illustrate cropped scannet_v2 data
  import glob
  fns = glob.glob('/home/ran/mesh_walker/datasets/scannet_v2_4cm_crops/*crop*.npz')
  fn = np.random.choice(fns)
  import dataset
  mesh = dataset.load_model_from_npz(fn)
  # set face colors according to vertex labels
  from scannet_preprocess.dataset_prepare_scannet_v2 import scannet_v2_colors
  visualize_model(mesh['vertices'], mesh['faces'], face_colors=[scannet_v2_colors[int(x)]  for x in mesh['labels']])