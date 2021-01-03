import os, shutil, time, copy, json, glob, pickle, glob, sys
from easydict import EasyDict
from tqdm import tqdm

import scipy
import numpy as np
import pylab as plt
import trimesh, open3d
from sklearn.manifold import TSNE

import tensorflow as tf

import rnn_model
import dataset
import dataset_prepare
import utils

USED_FIXED_LABEL_FOR_SHREC15_HUMAN = 0


def fill_edges(model):
  # To compare accuracies to MeshCNN, this function build edges & edges length in the same way they do
  edge2key = dict()
  edges_length = []
  edges = []
  edges_count = 0
  for face_id, face in enumerate(model['faces']):
    faces_edges = []
    for i in range(3):
      cur_edge = (face[i], face[(i + 1) % 3])
      faces_edges.append(cur_edge)
    for idx, edge in enumerate(faces_edges):
      edge = tuple(sorted(list(edge)))
      faces_edges[idx] = edge
      if edge not in edge2key:
        edge2key[edge] = edges_count
        edges.append(list(edge))
        e_l = np.linalg.norm(model['vertices'][edge[0]] - model['vertices'][edge[1]])
        edges_length.append(e_l)
        edges_count += 1
  model['edges_meshcnn'] = np.array(edges)
  model['edges_length'] = edges_length


def get_model_by_name(name):
  fn = name[name.find(':')+1:]
  mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
  model = {'vertices': mesh_data['vertices'], 'faces': mesh_data['faces'], 'labels': mesh_data['labels'],
           'edges': mesh_data['edges']}

  if 'face_labels' in mesh_data.keys():
     model['face_labels'] = mesh_data['face_labels']

  if 'labels_fuzzy' in mesh_data.keys():
    model['labels_fuzzy'] = mesh_data['labels_fuzzy']
    fill_edges(model)
    model['seseg'] = np.zeros((model['edges_meshcnn'].shape[0], model['labels_fuzzy'].shape[1]))
    for e in range(model['edges_meshcnn'].shape[0]):
      v0, v1 = model['edges_meshcnn'][e]
      l0 = model['labels_fuzzy'][v0]
      l1 = model['labels_fuzzy'][v1]
      model['seseg'][e] = (l0 + l1) / 2
  else:
    # To remove:
    seseg_fn = None
    if fn.find('/coseg/') != -1:
      n = fn.split('/')[-1].split('_')[1]
      part = name.split(':')[0]
      seseg_fn = os.path.expanduser('~') + '/datasets_processed/coseg/' + part + '/sseg/' + n + '.seseg'
    if fn.find('test_shrec__') != -1:
      n = fn.split('/')[-1].split('__')[-1].split('_')[0]
      seseg_fn = os.path.expanduser('~') + '/datasets_processed/human_seg/sseg/shrec__' + n + '.seseg'
    if seseg_fn is not None:
      model['seseg'] = np.loadtxt(seseg_fn)
      fill_edges(model)

  # Get original mesh & labels, only for sig17_seg_benchmark dataset
  # ----------------------------------------------------------------
  # Disabled, to be virified correctness
  if 1 and 'shrec' in fn:
    idx = fn.split('__shrec_')[-1].split('_')[0]
    if not idx.isdecimal():
      idx = fn.split('_shrec__')[-1].split('_')[0]
    if idx == '12':
      o_fn = '12_fix_orientation.off'
    else:
      o_fn = idx + '.off'
    o_mesh = trimesh.load_mesh(os.path.expanduser('~') + '/mesh_walker/datasets_raw/sig17_seg_benchmark/meshes/test/shrec/' + o_fn, process=False)
    if idx == '15' and USED_FIXED_LABEL_FOR_SHREC15_HUMAN:
      o_l_fn = 'shrec_' + idx + '_full.txt.fixed.txt'
    else:
      o_l_fn = 'shrec_' + idx + '_full.txt'
    o_face_labels = np.loadtxt(os.path.expanduser('~') + '/mesh_walker/datasets_raw/sig17_seg_benchmark/segs/test/shrec/' + o_l_fn)

    # Edges prepare
    e_labels = {}
    for f_idx, f in enumerate(o_mesh.faces):
      for e_i in [(0, 1), (1, 2), (2, 0)]:
        e_v = tuple(np.sort(f[np.array(e_i)]).tolist())
        e_label = o_face_labels[f_idx]
        if e_v not in e_labels.keys():
          e_labels[e_v] = [e_label]
        else:
          e_labels[e_v].append(e_label)

    model['original'] = {'o_faces': o_mesh.faces, 'o_vertices': o_mesh.vertices, 'o_face_labels': o_face_labels,
                         'o_area_faces': o_mesh.area_faces, 'e_labels': e_labels}

  return model


def calc_final_accuracy(models, print_details):
  # Calculating 6 types of accuracy.
  # 3 alternatives for element used (vertex / edge / face) and for each element, vanilla accuracy and normalized one.
  # Notes:
  # 1. For edge calculation only, the accuracy allow fuzzy labeling:
  #    like MeshCNN's paper, if an edge is inbetween two different segments, any prediction from the two is considered good.
  # 2. Normalized accuracy is calculated using the edge length or face area or vertex area (which is the mean faces area for each vertex).
  if print_details:
    print(utils.color.BOLD + utils.color.BLUE + '\n\nAccuracy report : ' + utils.color.END)
  vertices_accuracy = []; vertices_norm_acc = []
  faces_accuracy = []; faces_norm_acc = []
  faces_accuracy_orig = []; faces_norm_acc_orig = []
  edges_accuracy_orig = []
  edges_accuracy = []; edges_norm_acc = []
  ii = 0
  n_total_vertices = 0
  n_vertices_no_prediction = 0
  for model_name, model in models.items():
    if model['labels'].size == 0:
      continue
    n_vertices_no_prediction += np.sum((model['pred'].sum(axis=1) == 0))
    n_total_vertices += model['pred'].shape[0]
    best_pred = np.argmax(model['pred'], axis=-1)
    model['v_pred'] = best_pred
    pred_score = scipy.special.softmax(model['pred'], axis=1)
    # Calc edges accuracy
    if 'edges_meshcnn' in model.keys(): # pred per edge
      g = 0
      gn = 0
      for ei, edge in enumerate(model['edges_meshcnn']):
        v0_pred = best_pred[edge[0]]
        v0_score = pred_score[edge[0], v0_pred]
        v1_pred = best_pred[edge[1]]
        v1_score = pred_score[edge[1], v1_pred]
        if v0_score > v1_score:
          best = v0_pred - 1
        else:
          best = v1_pred - 1
        if best < model['seseg'].shape[1]:
          g  += (model['seseg'][ei, best] != 0)
          gn += (model['seseg'][ei, best] != 0) * model['edges_length'][ei]
      this_accuracy = g / model['edges_meshcnn'].shape[0]
      norm_accuracy = gn / np.sum(model['edges_length'])
      edges_accuracy.append(this_accuracy)
      edges_norm_acc.append(norm_accuracy)

    # Calc vertices accuracy
    if 'area_vertices' not in model.keys():
      dataset_prepare.calc_mesh_area(model)
    this_accuracy = (best_pred == model['labels']).sum() / model['labels'].shape[0]
    norm_accuracy = np.sum((best_pred == model['labels']) * model['area_vertices']) / model['area_vertices'].sum()
    vertices_accuracy.append(this_accuracy)
    vertices_norm_acc.append(norm_accuracy)

    # Calc faces accuracy
    if 'face_labels' in model.keys():
      face_preds = []
      for face in model['faces']:
        v_pred = model['pred'][face, :]
        v_pred_softmax = scipy.special.softmax(v_pred * 4 - 2, axis=1)
        face_pred_soft = np.mean(v_pred_softmax, axis=0)
        face_preds.append(np.argmax(face_pred_soft))
      model['f_pred'] = face_preds
      this_accuracy = np.mean(face_preds == model['face_labels'])
      norm_accuracy = np.sum((face_preds == model['face_labels']) * model['area_faces']) / model['area_faces'].sum()
      faces_accuracy.append(this_accuracy)
      faces_norm_acc.append(norm_accuracy)

      if 'original' in model.keys(): # project results to the original mesh (before remeshing)
        o_faces = model['original']['o_faces']
        o_vertices = model['original']['o_vertices']
        o_gt_labels = model['original']['o_face_labels']
        o_area_faces =  model['original']['o_area_faces']
        model['original']['o_face_preds'] = []
        o_face_preds = model['original']['o_face_preds']
        o_edge_pred = []
        simp_mesh = trimesh.Trimesh(model['vertices'], model['faces'], process=False)
        prox_q = trimesh.proximity.ProximityQuery(simp_mesh)
        for f_i, f in enumerate(o_faces):
          if 0:     # Take closest vertex prediction
            f_center = np.mean(o_vertices[f, :], axis=0)
            closest_vertex = prox_q.vertex([f_center])[1][0]
            o_face_preds.append(best_pred[closest_vertex])
          else:     # Take N closest vertices, and average predictions
            f_center = np.mean(o_vertices[f, :], axis=0)
            ds, closest_vertices = simp_mesh.kdtree.query(f_center, 6)
            if 0:
              pred = np.argmax(np.bincount(best_pred[closest_vertices]))
            else:
              tmp = np.zeros((model['pred'].shape[1],))
              ds = ds - np.min(ds)
              ds /= np.max(ds)
              for d, p in zip(ds, best_pred[closest_vertices]):
                tmp[p] += 1 - d
              pred = np.argmax(tmp)
            o_face_preds.append(pred)

          for e_i in [(0, 1), (1, 2), (2, 0)]:
            e_v = np.sort(f[np.array(e_i)])
            closest_vertices = prox_q.vertex([o_vertices[e_v, :]])[1][0]
            v0_pred = best_pred[closest_vertices[0]]
            v0_score = pred_score[closest_vertices[0], v0_pred]
            v1_pred = best_pred[closest_vertices[1]]
            v1_score = pred_score[closest_vertices[1], v1_pred]
            if v0_score > v1_score:
              best = v0_pred
            else:
              best = v1_pred
            gt = model['original']['e_labels'][tuple(e_v.tolist())]
            o_edge_pred.append(best in gt)

        o_face_preds = np.array(o_face_preds)
        model['original']['o_face_preds'] = o_face_preds

        if 1: # Fix zig-zag
          faces2change = []
          if 'f_nbr' in model['original'].keys():
            f_nbr = model['original']['f_nbr']
          else:
            o_mesh = trimesh.Trimesh(o_vertices, o_faces, process=False)
            face_adjacency = o_mesh.face_adjacency
            face_adjacency_angles = o_mesh.face_adjacency_angles
            model['original']['f_nbr'] = [[] for _ in range(o_faces.shape[0])]
            model['original']['f_nbr_ang'] = [[] for _ in range(o_faces.shape[0])]
            f_nbr = model['original']['f_nbr']
            f_nbr_ang = model['original']['f_nbr_ang']
            for adj, ang in zip(face_adjacency, face_adjacency_angles):
              f_nbr[adj[0]].append(adj[1])
              f_nbr[adj[1]].append(adj[0])
              f_nbr_ang[adj[0]].append(ang)
              f_nbr_ang[adj[1]].append(ang)
          for f_idx, f_nbrs in enumerate(f_nbr):
            this_pred = o_face_preds[f_idx]
            nbrs_pred = o_face_preds[f_nbrs]
            if (nbrs_pred != this_pred).sum() >= 1:
              faces2change.append(f_idx)
          for f2c in faces2change:
            o_face_preds[f2c] = -1
          g_change = 0
          b_change = 0
          n_faces2change = len(faces2change)
          while faces2change:
            this_group = []
            this_group_pred = np.zeros((model['pred'].shape[1],))
            f_ = faces2change.pop()
            this_group.append(f_)
            nbrs = f_nbr[f_]
            nbrs_angs = f_nbr_ang[f_]
            while nbrs:
              n = nbrs.pop()
              ang = nbrs_angs.pop()
              if o_face_preds[n] == -1: # Also not sure
                this_group.append(n)
                for nbr, an in zip(f_nbr[n], f_nbr_ang[n]):
                  if nbr not in this_group and nbr not in nbrs:
                    nbrs.append(nbr)
                    nbrs_angs.append(an)
              else:
                this_group_pred[o_face_preds[n]] += 180 - 180 * ang / np.pi
            for f2c in this_group:
              new_pred = np.argmax(this_group_pred)
              g_change += new_pred == o_gt_labels[f2c] and new_pred != o_face_preds[f2c]
              b_change += new_pred != o_gt_labels[f2c] and new_pred != o_face_preds[f2c]
              o_face_preds[f2c] = new_pred
              if f2c in faces2change:
                faces2change.remove(f2c)
          print('  ==> zig-zag: n faces changes ', n_faces2change, ' ; ', round(100 * n_faces2change / o_faces.shape[0], 1), '%',
                g_change, b_change)

        accuracy_on_orig_mesh = np.mean(o_face_preds == o_gt_labels)
        accuracy_on_orig_mesh_normed = np.sum((o_face_preds == o_gt_labels) * o_area_faces) / o_area_faces.sum()
        print('--- accuracy_on_orig_mesh', round(accuracy_on_orig_mesh * 100, 1))
        print('--- o_edge_pred', np.mean(o_edge_pred))
        edges_accuracy_orig.append(np.mean(o_edge_pred))
        faces_accuracy_orig.append(accuracy_on_orig_mesh)
        faces_norm_acc_orig.append(accuracy_on_orig_mesh_normed)

    we_have_results = len(edges_accuracy) > 0 and len(vertices_accuracy) > 0
    if print_details and we_have_results:
      print('  ', ii, ') ', model_name.split('/')[-1], ':', np.round(edges_accuracy[-1] * 100, 1), '%', 'normed: ', np.round(edges_norm_acc[-1] * 100, 1), '%')
      print('                              vertex analysis :', np.round(vertices_accuracy[-1] * 100, 1), '%', 'normed: ', np.round(vertices_norm_acc[-1] * 100, 1), '%')
    ii += 1
  if print_details:
    if len(edges_norm_acc):
      print('-------------\n')
      print('Edges:\n  Average accuracy: ', np.round(np.mean(edges_accuracy) * 100, 2), '%')
      print('  Normed accuracy : ', np.round(np.mean(edges_norm_acc) * 100, 2), '%')
      print('Vertices:\n  Average accuracy: ', np.round(np.mean(vertices_accuracy) * 100, 2), '%')
      print('  Normed accuracy : ', np.round(np.mean(vertices_norm_acc) * 100, 2), '%')
      print('Faces:\n  Average accuracy: ', np.round(np.mean(faces_accuracy) * 100, 2), '%')
      print('  Normed accuracy : ', np.round(np.mean(faces_norm_acc) * 100, 2), '%')
      print('\n\n\n')
      print('% Visited vertices:', round((n_total_vertices - n_vertices_no_prediction) / n_total_vertices * 100, 1),
            'n_vertices_no_prediction', n_vertices_no_prediction, 'n_total_vertices', n_total_vertices)
      if len(faces_accuracy_orig):
        print('Faces-orig mesh:\n  Average accuracy: ', np.round(np.mean(faces_accuracy_orig) * 100, 2), '%')
        print('  Normed accuracy : ', np.round(np.mean(faces_norm_acc_orig) * 100, 2), '%')
        print('  edges_accuracy_orig : ', np.round(np.mean(edges_accuracy_orig) * 100, 2), '%')
    else:
      print('Accuracy not calculated (no labels)')

  if len(edges_accuracy) == 0:
    edges_accuracy = [0]
  if len(faces_accuracy) == 0:
    faces_accuracy = [0]

  return np.mean(edges_accuracy), np.mean(vertices_accuracy), np.mean(faces_accuracy)


def postprocess_vertex_predictions(models):
  PAPER_POST_PROCESS = False
  for model_name, model in models.items():
    pred_orig = model['pred'].copy()
    av_pred = np.zeros_like(pred_orig)
    for v in range(model['vertices'].shape[0]):
      if PAPER_POST_PROCESS:
        this_pred = pred_orig[v]
      else:
        this_pred = pred_orig[v] / model['pred_count'][v]
      nbrs_ids = model['edges'][v]
      nbrs_ids = np.array([n for n in nbrs_ids if n != -1])
      if nbrs_ids.size:
        if PAPER_POST_PROCESS:
          first_ring_pred = pred_orig[nbrs_ids]
        else:
          first_ring_pred = (pred_orig[nbrs_ids].T / model['pred_count'][nbrs_ids]).T
        nbrs_pred = np.mean(first_ring_pred, axis=0) * 0.5
        av_pred[v] = this_pred + nbrs_pred
      else:
        av_pred[v] = this_pred
    model['pred'] = av_pred


def calc_accuracy_test(dataset_folder=False, logdir=None, dnn_model=None, params=None, verbose_level=2,
                       n_iters=None, dump_model=False, min_max_faces2use=[0, np.inf], model_fn=None, n_target_faces=None,
                       n_walks_per_model=32, seq_len=None, params2overide={}, data_augmentation={}):
  print_details = verbose_level >= 2
  if params is None:
    with open(logdir + '/params.txt') as fp:
      params = EasyDict(json.load(fp))
    params.model_fn = logdir + '/learned_model.keras'
    params.new_run = 0
  else:
    params = copy.deepcopy(params)
  params.mix_models_in_minibatch = False
  params.batch_size = 1
  params.net_input.append('vertex_indices')
  if n_iters is None:
    n_iters = 32
  if seq_len:
    params.seq_len = seq_len
  if 1:
    walks_f = 1
  else:
    walks_f = n_iters
    n_iters = 1
  params.n_walks_per_model = n_walks_per_model * walks_f
  for k, v in params2overide.items():
    params[k] = v

  if print_details and model_fn is not None:
    print(utils.color.BOLD + utils.color.BLUE + 'logdir : ', model_fn + utils.color.END)

  params.net_input.append('vertex_indices')

  if dataset_folder.split('.')[-1] in ['off', 'ply', 'obj']:
    dataset_output_path = os.path.expanduser('~') + '/mesh_walker/datasets_processed/tmp'
    dataset_prepare.create_tmp_dataset(dataset_folder, dataset_output_path, n_target_faces=n_target_faces)
    pathname_expansion = dataset_output_path + '/*.npz'
  else:
    pathname_expansion = dataset_folder
  test_dataset, n_items = dataset.tf_mesh_dataset(params, pathname_expansion, mode=params.network_task,
                                                  shuffle_size=0, size_limit=np.inf, permute_file_names=False,
                                                  min_max_faces2use=min_max_faces2use, must_run_on_all=True,
                                                  min_dataset_size=1, data_augmentation=data_augmentation)

  if dnn_model is None:
    dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim - 1, model_fn, model_must_be_load=True,
                                       dump_model_visualization=False)

  skip = int(params.seq_len * 0.5)
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  n_walks = 0
  models = {}
  for iter in tqdm(range(n_iters), disable=print_details):
    i_m = 0
    for name_, model_ftrs_, labels_ in test_dataset:
      name = name_.numpy()[0].decode()
      i_m += 1
      assert name_.shape[0] == 1
      if 0:       # Test scale augmentations
        m = 0.5
        scale_augmentation = np.random.uniform(1 - m, 1 + m)
      else:
        scale_augmentation = 1
      model_ftrs = model_ftrs_[:, :, :, :-1] * scale_augmentation
      all_seq = model_ftrs_[:, :, :, -1].numpy()
      n_walks += model_ftrs.shape[0]
      if name not in models.keys():
        models[name] = get_model_by_name(name)
        models[name]['pred'] = np.zeros((models[name]['vertices'].shape[0], params.n_classes))
        models[name]['pred_count'] = 1e-6 * np.ones((models[name]['vertices'].shape[0], )) # Initiated to a very small number to avoid devision by 0

      sp = model_ftrs.shape
      ftrs = tf.reshape(model_ftrs, (-1, sp[-2], sp[-1]))
      labels = tf.reshape(labels_, (-1, sp[-2]))
      strd = int(params.n_walks_per_model / walks_f)
      predictions = None
      for g in range(walks_f):
        predictions_ = dnn_model(ftrs[g * strd:(g + 1) * strd], training=False).numpy()[:, skip:]
        if predictions is None:
          predictions = np.zeros((params.n_walks_per_model, predictions_.shape[1], predictions_.shape[2]))
        predictions[g * strd:(g + 1) * strd] = predictions_
      all_seq = all_seq[0, :, skip + 1:].reshape(-1).astype(np.int32)
      predictions4vertex = predictions.reshape((-1, predictions.shape[-1]))
      labels = labels[:, skip + 1:]
      for w_step in range(all_seq.size):
        models[name]['pred'][all_seq[w_step]] += predictions4vertex[w_step]
        models[name]['pred_count'][all_seq[w_step]] += 1

      test_accuracy(labels, predictions)
      if print_details:
        print(iter, '-', n_walks, ':', test_accuracy.result().numpy())

  #calc_final_accuracy(models, print_details)
  #postprocess_vertex_predictions(models)
  e_acc_after_postproc, v_acc_after_postproc, f_acc_after_postproc = calc_final_accuracy(models, print_details)

  # For visualization (using meshlab), dump the resuls to ply files
  if dump_model:
    for colorize_only_bad_pred in [0, 1]:
      for name, model in tqdm(models.items()):
        if 0:
          utils.visualize_model(model['vertices'], model['faces'], v_size=1, show_vertices=0, face_colors=model['face_labels'], show_edges=0)
          utils.visualize_model(model['vertices'], model['faces'], v_size=1, show_vertices=0, face_colors=model['f_pred'], show_edges=0)
          utils.visualize_model(model['vertices'], model['faces'], vertex_colors_idx=model['labels'].astype(np.int), point_size=5, show_edges=0)
          utils.visualize_model(model['vertices'], model['faces'], vertex_colors_idx=model['v_pred'].astype(np.int), point_size=5, show_edges=0)
        pred_per_node = model['pred']
        no_pred_idxs = np.where(np.abs(pred_per_node).sum(axis=1) == 0)[0]
        best_per_node = np.argmax(pred_per_node, axis=1)
        colors_parts = np.zeros((model['vertices'].shape[0], 3))
        if 0:
          wrong_pred = np.ones_like(best_per_node)
          for v in range(best_per_node.shape[0]):
            wrong_pred[v] = (model['labels4test'][v][best_per_node[v] - 1] == 0)
        else:
          wrong_pred = best_per_node != model['labels']
        for part_id in range(16):
          if colorize_only_bad_pred:
            idxs = np.where((best_per_node == part_id) * wrong_pred)
          else:
            idxs = np.where(best_per_node == part_id)
          colors_parts[idxs] = utils.index2color(part_id)
        colors_parts[no_pred_idxs] = [0, 0, 0]
        v_clrs = colors_parts

        model = EasyDict({'vertices': models[name]['vertices'], 'faces': models[name]['faces']})
        model_fn = name.split('.')[-2].split('/')[-1]
        model_fn += '_only_bad_pred' if colorize_only_bad_pred else '_all'
        db_name = name[:name.find(':')]
        utils.colorize_and_dump_model(model, [], 'debug_models/eval_' + db_name + '_' + model_fn + '.ply',
                                        vertex_colors=v_clrs, norm_clrs=0, show=False, verbose=False)
        if 'original' in models[name].keys():
          o_model = EasyDict({'vertices': models[name]['original']['o_vertices'], 'faces': models[name]['original']['o_faces']})
          o_f_colors = np.zeros((o_model['faces'].shape[0], 3))
          for i, label in enumerate(models[name]['original']['o_face_labels']):
            pred = models[name]['original']['o_face_preds'][i]
            if not colorize_only_bad_pred or pred != label:
              o_f_colors[i] = utils.index2color(pred)
          utils.colorize_and_dump_model(o_model, [], 'debug_models/eval_' + db_name + '_' + model_fn + '_orig.ply',
                                        clrs=o_f_colors, norm_clrs=0, show=False, verbose=False)
  return [f_acc_after_postproc, e_acc_after_postproc], dnn_model


def show_coseg_chair():
  n_iters = 4 # for fast run set : n_iters = 2

  n_target_faces = None
  logdir = '/home/alonlahav/mesh_walker/runs_test/0045-08.07.2020..14.44__coseg_chairs/'
  model_fn = logdir + 'learned_model2keep__00060000.keras'
  dataset2use = '/home/alonlahav/mesh_walker/remesh_check/coseg_chair_0014_1000f.obj'
  dataset2use = '/home/alonlahav/mesh_walker/remesh_check/coseg_chair_remeshed_12kFaces.ply'
  n_target_faces = 1000

  accs, dnn_model = calc_accuracy_test(logdir=logdir, dataset_folder=dataset2use, n_iters=n_iters, dump_model=True, n_target_faces=n_target_faces, model_fn=model_fn)

  exit(0)


def coseg_stat():
  n_iters = 1
  if 0:
    logdir = '/home/alonlahav/mesh_walker/runs_test/0045-08.07.2020..14.44__coseg_chairs'
    dataset2use = os.path.expanduser('~') + '/mesh_walker/datasets_processed/coseg_from_meshcnn/coseg_chairs/*test*.npz'
  if 0:
    logdir = '/home/alonlahav/mesh_walker/runs_test/0043-07.07.2020..20.29__coseg_aliens'
    dataset2use = os.path.expanduser('~') + '/mesh_walker/datasets_processed/coseg_from_meshcnn/coseg_aliens/*test*.npz'
  if 1:
    logdir = '/home/alonlahav/mesh_walker/runs_test/0044-08.07.2020..04.53__coseg_vases'
    dataset2use = os.path.expanduser('~') + '/mesh_walker/datasets_processed/coseg_from_meshcnn/coseg_vases/*test*.npz'
  calc_accuracy_test(logdir=logdir, dataset_folder=dataset2use, n_iters=n_iters, dump_model=False)
  exit(0)

def check_rotation_weak_points():
  n_iters = 32
  logdir = os.path.expanduser('~') + '/mesh_walker/'

  ds = 'sig17_seg_benchmark-1.5k'     #'sig17_seg_benchmark-6k'
  if 1:
    logdir += 'runs_aug_360/0010-08.08.2020..23.54__human_seg-1.5k_from_meshcnn/'
  else:
    logdir += 'runs_aug_45/0010-08.08.2020..23.51__human_seg-1.5k_from_meshcnn/'
  model_fn = logdir + 'learned_model2keep__00060005.keras'

  dataset2use = os.path.expanduser('~') + '/mesh_walker/datasets_processed/' + ds + '/*test*.npz'
  #dataset2use = os.path.expanduser('~') + '/mesh_walker/datasets_processed/' + ds + '/test__shrec_2*.npz'
  dnn_model = None

  rot_angles = range(0, 360, 10)
  for axis in [0, 1, 2]:
    accs = []
    dataset.data_augmentation_rotation.test_rotation_axis = axis
    for rot in rot_angles:
      acc, dnn_model = calc_accuracy_test(logdir=logdir, dataset_folder=dataset2use, n_iters=n_iters, dump_model=False,
                                           model_fn=model_fn, verbose_level=0, dnn_model=dnn_model, data_augmentation={'rotation': rot})
      accs.append(acc[0])
      print(rot, accs)
    plt.plot(rot_angles, accs)
    plt.xlabel('Rotation [degrees]')
    plt.ylabel('Accuracy')
    plt.title('Accuracy VS rotation angles')
  plt.legend(['axis=0', 'axis=1', 'axis=2'])
  plt.suptitle('/'.join(logdir.split('/')[-3:-1]))
  plt.show()



if __name__ == '__main__':
  utils.config_gpu(1)
  np.random.seed(0)
  tf.random.set_seed(0)

  if 0:
    check_rotation_weak_points()
    exit(0)

  #coseg_stat()

  n_iters = 2 # for fast run set : n_iters = 2

  n_target_faces = None
  logdir = os.path.expanduser('~') + '/mesh_walker/'

  ds = 'sig17_seg_benchmark-no_simplification'
  #ds = 'sig17_seg_benchmark-6k'
  logdir += 'runs_aug_360/0002-14.08.2020..16.07__human_seg-no_smpl/'
  model_fn = None # logdir + 'learned_model2keep__00010010.keras'

  if 1: # Run on all test meshes
    dataset2use = os.path.expanduser('~') + '/mesh_walker/datasets_processed/' + ds + '/*test*.npz'
    dataset2use = os.path.expanduser('~') + '/mesh_walker/datasets_processed/' + ds + '/test__shrec_15*.npz'
  else: # Run on one mesh, only for visualize
    verbose_level = 0
    mesh_id = '183' # 180 / 100 / 008 / 016 / 018
    dataset2use = '/home/alonlahav/datasets_processed/MPI-FAUST/test/scans/test_scan_' + mesh_id + '.ply'
    n_target_faces = 4000

  params2overide = EasyDict()
  #params2overide.n_target_vrt_to_norm_walk = 3000

  calc_accuracy_test(logdir=logdir, dataset_folder=dataset2use, n_iters=n_iters, dump_model=True, n_target_faces=n_target_faces,
                     model_fn=model_fn, params2overide=params2overide)

