import os, shutil, time, copy, glob
from easydict import EasyDict
import json
import platform

import numpy as np
import tensorflow as tf
import trimesh, open3d
import pyvista as pv
import scipy
import pylab as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from tqdm import tqdm

import rnn_model
import utils
import dataset
import dataset_prepare
from evaluation import calculate_map_auc

timelog = {}
timelog['prep_model'] = []
timelog['fill_features'] = []

def get_model_names():
  part = 'test'
  model_fns = []
  for i, name in enumerate(dataset_prepare.model_net_labels):
    pathname_expansion = os.path.expanduser('~') + '/datasets/ModelNet40/' + name + '/' + part + '/*.off'
    filenames = glob.glob(pathname_expansion)
    model_fns += filenames
  return model_fns

def show_walk(model, features, one_walk=False, weights=False, save_name=''):
  if weights:
    walks = features[:5,:,-1]
    rendered = utils.visualize_model(model['vertices'], model['faces'], walk=list(walks.astype(np.int32)), off_screen=True)

    # TODO: save rendered to file
  else:
    for wi in range(features.shape[0]):
      walk = features[wi, :, -1].astype(np.int)
      jumps = features[wi, :, -2].astype(np.bool)
      utils.visualize_model_walk(model['vertices'], model['faces'], walk, jumps)
      if one_walk:
        break


def calc_accuracy_test(dataset_folder=False, logdir=None, labels=None, iter2use='last', classes_indices_to_use=None,
                       dnn_model=None, params=None, verbose_level=2, min_max_faces2use=[0, 4000], model_fn=None,
                       target_n_faces=['according_to_dataset'], n_walks_per_model=16, seq_len=None, data_augmentation={}):
  SHOW_WALK = 0
  SHOW_WEIGHTS = 0
  WALK_LEN_PROP_TO_NUM_OF_TRIANLES = 0
  COMPONENT_ANALYSIS = False

  np.random.seed(1)
  tf.random.set_seed(0)
  classes2use = None #['desk', 'dresser', 'table', 'laptop', 'lamp', 'stool', 'wardrobe'] # or "None" for all
  print_details = verbose_level >= 2
  if params is None:
    with open(logdir + '/params.txt') as fp:
      params = EasyDict(json.load(fp))
    if model_fn is not None:
      pass
    elif iter2use != 'last':
      model_fn = logdir + '/learned_model2keep--' + iter2use
      model_fn = model_fn.replace('//', '/')
    else:
      model_fn = tf.train.latest_checkpoint(logdir)
    if verbose_level and model_fn is not None:
      print(utils.color.BOLD + utils.color.BLUE + 'logdir : ', model_fn + utils.color.END)
  else:
    params = copy.deepcopy(params)
  params.batch_size = 1
  params.n_walks_per_model = n_walks_per_model
  # params.classes_indices_to_use = [x for x in range(params.n_classes) if x not in [19, 30]]


  params.classes_indices_to_use = None
  if seq_len:
    params.seq_len = seq_len
  if verbose_level:
    print('params.seq_len:', params.seq_len, ' ; n_walks_per_model:', n_walks_per_model)

  if SHOW_WALK or SHOW_WEIGHTS:
    params.net_input += ['vertex_indices']

  params.set_seq_len_by_n_faces = 1
  if dataset_folder:
    size_limit = np.inf # 200
    params.classes_indices_to_use = classes_indices_to_use
    pathname_expansion = dataset_folder
    if hasattr(params, 'retrieval_80_20'):

      # for each class, randomly choose 20 test models, except the 80 taken for training.
      test_dataset, n_models_to_test = dataset.tf_mesh_dataset(params, pathname_expansion, mode=params.network_task,
                                                                shuffle_size=0, size_limit=size_limit, permute_file_names=True,
                                                                min_max_faces2use=min_max_faces2use, must_run_on_all=True,
                                                                data_augmentation=data_augmentation)
    else:
      test_dataset, n_models_to_test = dataset.tf_mesh_dataset(params, pathname_expansion, mode=params.network_task,
                                                                shuffle_size=0, size_limit=size_limit, permute_file_names=True,
                                                                min_max_faces2use=min_max_faces2use, must_run_on_all=True,
                                                                data_augmentation=data_augmentation)
  else:
    test_dataset = get_model_names()
    test_dataset = np.random.permutation(test_dataset)
    n_models_to_test = len(test_dataset)

  if dnn_model is None:
    if params.net == 'HierTransformer':
      import attention_model
      dnn_model = attention_model.WalkHierTransformer(**params.net_params, params=params,
                                                      model_fn=model_fn, model_must_be_load=True)
    else:
      dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim - (SHOW_WALK or SHOW_WEIGHTS), model_fn,
                                       model_must_be_load=True, dump_model_visualization=False)

  n_pos_all = 0
  n_classes = params.n_classes
  all_confusion = np.zeros((n_classes, n_classes), dtype=np.int)
  size_accuracy = []
  ii = 0
  tb_all = time.time()
  metrics= {}
  res_per_n_faces = {}
  pred_per_model_name = {}
  pred_per_model_per_faces = {}
  ftrs_per_model_name = {}
  dnn_inference_time = [] # 150mSec for 64 walks of 200 steps
  bad_pred = EasyDict({'n_comp': [], 'biggest_comp_area_ratio': []})
  good_pred = EasyDict({'n_comp': [], 'biggest_comp_area_ratio': []})
  all_features = []
  all_labels = []
  models_ids = []
  for i, data in tqdm(enumerate(test_dataset), disable=print_details, total=n_models_to_test):
    name, ftrs, gt = data
    model_fn = name.numpy()[0].decode()
    model_name, n_faces = utils.get_model_name_from_npz_fn(model_fn)
    assert ftrs.shape[0] == 1, 'Must have one model per batch for test'
    if WALK_LEN_PROP_TO_NUM_OF_TRIANLES:
      n2keep = int(n_faces / 2.5)
      ftrs = ftrs[:, :, :n2keep, :]
    ftrs = tf.reshape(ftrs, ftrs.shape[1:])
    gt = gt.numpy()[0]
    predictions = None
    for i_f, this_target_n_faces in enumerate(target_n_faces):
      model = None
      if SHOW_WALK:
        if model is None:
          model = dataset.load_model_from_npz(model_fn)
        if model['vertices'].shape[0] < 1000:
          print(model_fn)
          print('nv: ', model['vertices'].shape[0])
          show_walk(model, ftrs.numpy(), one_walk=1)

      ftr2use = ftrs[:, :, :-1].numpy() if SHOW_WALK or SHOW_WEIGHTS else ftrs.numpy()
      tb = time.time()
      if 0:
        jumps = ftr2use[:,:,3]
        jumps = np.hstack((jumps, np.ones((jumps.shape[0], 1)))) # To promise that one jump is found
        ftr2use = ftr2use[:,:,:3]
        first_jumps = [np.where(j)[0][0]  for j in jumps]
        last_jumps  = [0,0,0] # [np.where(j)[0][-2] - np.where(j)[0][-3] for j in jumps]
        plt.hist(first_jumps)
        plt.hist(last_jumps)
        model = dataset.load_model_from_npz(model_fn)
        plt.title('#vertices / faces : ' + str(model['vertices'].shape[0]) + ' / ' + str(model['faces'].shape[0]))
        plt.show()
      if dnn_model.cross_walk_attn:
        predictions_, weights, features = [x.numpy() for x in dnn_model(ftr2use, classify='weights', training=False)]
      else:
        predictions_, features = [x.numpy() for x in dnn_model(ftr2use, classify='both', training=False)]
      te = time.time() - tb
      dnn_inference_time.append(te / n_walks_per_model * 1000)
      if 0:#len(dnn_inference_time) == 10:
        print(dnn_inference_time)
        plt.hist(dnn_inference_time[1:])
        plt.xlabel('[mSec]')
        plt.show()
      if predictions is None:
        predictions = predictions_
      else:
        predictions = np.vstack((predictions, predictions_))

      all_labels.append(gt)
      all_features.append(np.mean(features, axis=0) / np.linalg.norm(np.mean(features, axis=0)))
      # all_features.append(np.mean(predictions_, axis=0) / np.linalg.norm(np.mean(predictions_, axis=0)))
      models_ids.append(model_name)
      if SHOW_WEIGHTS:
        # TODO: show only weights of walks where Alon's model failed
        # Showing walks with weighted attention - which walks recieved higher weights
        sorted_weights = np.argsort(weights.squeeze())[::-1]
        sorted_features = ftrs.numpy()[sorted_weights]
        if model is None:
          model = dataset.load_model_from_npz(model_fn)
        print(model_fn)
        print('nv: ', model['vertices'].shape[0])
        show_walk(model, sorted_features, weights=True, save_name=model_fn)

    mean_pred = np.mean(predictions, axis=0)
    max_hit = np.argmax(mean_pred)

    if model_name not in pred_per_model_name.keys():
      pred_per_model_name[model_name] = [gt, np.zeros_like(mean_pred)]
    pred_per_model_name[model_name][1] += mean_pred

    mean_features = np.mean(features, axis=0) / np.linalg.norm(np.mean(features, axis=0))
    if model_name not in ftrs_per_model_name.keys():
      ftrs_per_model_name[model_name] = {n_faces: mean_features}
    else:
      ftrs_per_model_name[model_name][n_faces] = mean_features

    # for logging errors
    if model_name not in pred_per_model_per_faces.keys():
      pred_per_model_per_faces[model_name] = {n_faces: mean_pred}
    else:
      pred_per_model_per_faces[model_name][n_faces] = mean_pred


    str2add = '; n.unique models: ' + str(len(pred_per_model_name.keys()))

    if n_faces not in res_per_n_faces.keys():
      res_per_n_faces[n_faces] = [0, 0]
    res_per_n_faces[n_faces][0] += 1

    if COMPONENT_ANALYSIS:
      model = dataset.load_model_from_npz(model_fn)
      comp_summary = dataset_prepare.component_analysis(model['faces'], model['vertices'])
      comp_area = [a['area'] for a in comp_summary]
      n_components = len(comp_summary)
      biggest_comp_area_ratio = np.sort(comp_area)[-1] / np.sum(comp_area)

    if max_hit != gt:
      false_str = ' , predicted: ' + labels[int(max_hit)] + ' ; ' + model_fn
      if COMPONENT_ANALYSIS:
        bad_pred.n_comp.append(n_components)
        bad_pred.biggest_comp_area_ratio.append(biggest_comp_area_ratio)
    else:
      res_per_n_faces[n_faces][1] += 1
      false_str = ''
      if COMPONENT_ANALYSIS:
        good_pred.n_comp.append(n_components)
        good_pred.biggest_comp_area_ratio.append(biggest_comp_area_ratio)
    if print_details:
      print('  ', max_hit == gt, labels[int(gt)], false_str, 'n_vertices: ')#, model['vertices'].shape[0])
      if 0:#max_hit != gt:
        model = dataset.load_model_from_npz(model_fn)
        utils.visualize_model(model['vertices'], model['faces'], line_width=1, opacity=1)

    all_confusion[int(gt), max_hit] += 1
    n_pos_all += (max_hit == gt)
    ii += 1
    if print_details:
      print(i, '/', n_models_to_test, ')  Total accuracy: ', round(n_pos_all / ii * 100, 1), 'n_pos_all:', n_pos_all, str2add)

  # =========== Retrieval calculations =========== #
  # if not os.path.exists(os.path.join(params.logdir, 'results')):
  #   os.makedirs(os.path.join(params.logdir, 'results'))
  # with open(os.path.join(params.logdir, 'results/features_per_model.json'), 'w') as fp:
  #   json.dump(ftrs_per_model_name, fp)
  # retrieval_metrics = calculate_retrieval_multiscale(ftrs_per_model_name)
  models_ids =np.asarray(models_ids)
  m_ids = np.zeros(models_ids.shape).astype(np.int32)
  for k, v in enumerate(np.unique(models_ids)):
    m_ids[models_ids == v] = int(k)
  np.savez(os.path.join(params.logdir, 'features.npz'),
           features=np.stack(all_features, axis=0),
           labels=np.asarray(all_labels),
           model_ids=m_ids)
  metrics.update(calculate_map_auc(np.stack(all_features, axis=0), np.asarray(all_labels), m_ids))

  if print_details:
    print(utils.color.BLUE + 'Total time, all:', time.time() - tb_all, utils.color.END)
    for k,v in metrics.items():
      print('{}: {:2.4f}'.format(k,v))
  n_models = 0
  n_sucesses = 0
  all_confusion_all_faces = np.zeros((n_classes, n_classes), dtype=np.int)
  false_pred_model_names = []
  for k, v in pred_per_model_name.items():
    gt = v[0]
    pred = v[1]
    max_hit = np.argmax(pred)
    all_confusion_all_faces[gt, max_hit] += 1
    n_models += 1
    n_sucesses += max_hit == gt
    if max_hit != gt:
      false_pred_model_names.append(k)

  mean_accuracy_all_faces = n_sucesses / n_models
  if print_details:
    print('\n\n ---------------\nOn avarage, for all faces:')
    print('  Accuracy: ', np.round(mean_accuracy_all_faces * 100, 2), '% ; n models checkd: ', n_models)
    print('Results per number of faces:')
    print('  ', res_per_n_faces, '\n\n--------------\n\n')

  if 1:  #For false preds analysis per faces rigorous
    csv_rows = []
    for k in false_pred_model_names:
      if 'modelnet' in params.logdir:
        m_name = k.split('_')[0]
        if m_name in ['night', 'flower', 'tv', 'range', 'glass']:
          m_name += '_' + k.split('_')[1]
        gt = labels.index(m_name)
      elif 'shrec11' in params.logdir:
        m_name = k.split('_')[0]
        labels_file = '/home/ran/mesh_walker/datasets/shrec11/ID_to_labels.txt'
        labels_per_id = [x[:-1] for x in open(labels_file, 'r').readlines()]
        gt = int(labels_per_id[int(m_name[1:])])
        m_name = dataset_prepare.shrec11_labels[gt]
      elif 'future' in params.logdir:
        labels_file = '/home/ran/Databases/3D-FUTURE/GT/model_infos.json'
        with open(labels_file, 'r') as f:
          labels_per_id = {x['model_id']: x['category'] for x in json.load(f) if not x['is_train']}
        m_name = labels_per_id[k]
        gt = dataset_prepare.future3d_shape2label[m_name]
      else:
        continue

      csv_row = [k, m_name, np.sum(all_confusion_all_faces[gt])]
      n_faces_list = list(pred_per_model_per_faces[k].keys())
      n_faces_list.sort()
      for n_faces in n_faces_list:
        preds = pred_per_model_per_faces[k][n_faces]
        csv_row += [n_faces, '{:2.3f}'.format(preds.max()), labels[preds.argmax()], '{:2.3f}'.format(preds[gt]), m_name]
      # n_faces_orig = glob.glob('/home/ran/mesh_walker/datasets/modelnet40_nearest_faces/*{}*'.format(k))[0].split('_')[7]
      # csv_row += [n_faces_orig]
      csv_rows.append(csv_row)
    import csv
    fn = 'False_preds_{:4d}.csv'.format(int(mean_accuracy_all_faces*10000))
    with open(os.path.join(params.logdir, 'results', fn), 'w') as f:
      csvwriter = csv.writer(f)
      csvwriter.writerows(csv_rows)

  if 1:
    bins = [0, 200, 500, 1000, 1500, 2000, 3000, 5000]
    accuracy_per_n_faces = []
    for b in range(len(bins) - 1):
      ks = [k for k in res_per_n_faces.keys() if k > bins[b] and k <= bins[b + 1]]
      attempts = 0
      successes = 0
      for k in ks:
        attempts_, successes_ = res_per_n_faces[k]
        attempts += attempts_
        successes += successes_
      if attempts:
        accuracy_per_n_faces.append(successes / attempts)
      else:
        accuracy_per_n_faces.append(np.nan)
    x = (np.array(bins[1:]) + np.array(bins[:-1])) / 2
    plt.figure()
    plt.plot(x, accuracy_per_n_faces)
    plt.xlabel('Number of faces')
    plt.ylabel('Accuracy')
    plt.show()

  if 0:
    b = 0;
    e = 40
    utils.plot_confusion_matrix(all_confusion[b:e, b:e], labels[b:e], normalize=1, show_txt=0)

  # Print list of accuracy per model
  for confusion in [all_confusion, all_confusion_all_faces]:
    if print_details:
      print('------')
    acc_per_class = []
    for i, name in enumerate(labels):
      this_type = confusion[i]
      n_this_type = this_type.sum()
      accuracy_this_type = this_type[i] / n_this_type
      if n_this_type:
        acc_per_class.append(accuracy_this_type)
      this_type_ = this_type.copy()
      this_type_[i] = -1
      scnd_best = np.argmax(this_type_)
      scnd_best_name = labels[scnd_best]
      accuracy_2nd_best = this_type[scnd_best] / n_this_type
      if print_details:
        msg = str(i).ljust(3), name.ljust(12), n_this_type, ',', str(round(accuracy_this_type * 100, 1)).ljust(5), ' ; 2nd best:', scnd_best_name.ljust(12), round(accuracy_2nd_best * 100, 1)
        with open(os.path.join(params.logdir, 'acc_per_class.txt'), 'w') as f:
          msg = ''.join([str(x) for x in msg])
          f.write(msg + '\n')
        print(str(i).ljust(3), name.ljust(12), n_this_type, ',', str(round(accuracy_this_type * 100, 1)).ljust(5), ' ; 2nd best:', scnd_best_name.ljust(12), round(accuracy_2nd_best * 100, 1))
  mean_acc_per_class = np.mean(acc_per_class)
  metrics['class_mean_accuracy'] = mean_acc_per_class
  metrics['overall_accuracy'] = mean_accuracy_all_faces
  if 0:
    print('Time Log:')
    for k, v in timelog.items():
      print('  ' , k, ':', np.mean(v))

  return metrics, dnn_model

def show_features_tsne(from_cach_dataset, logdir, dnn_model=None, cls2show=None, n_iters=None, dataset_labels=None,
                       model_fn='', max_size_per_class=5):
  with open(logdir + '/params.txt') as fp:
    params = EasyDict(json.load(fp))
  params.network_task = 'classification'
  params.batch_size = 1
  params.one_label_per_model = True
  params.n_walks_per_model = 8
  params.logdir = logdir
  params.seq_len = 200
  params.new_run = 0
  if n_iters is None:
    n_iters = 1

  params.classes_indices_to_use = cls2show
  pathname_expansion = os.path.expanduser('~') + '/mesh_walker/datasets_processed/' + from_cach_dataset
  test_dataset, n_items = dataset.tf_mesh_dataset(params, pathname_expansion, mode=params.network_task,
                                                            max_size_per_class=max_size_per_class)

  if params.net == 'RnnWalkNet':
    print('RnnWalkNet')
    dnn_model = dnn_cad_seq.RnnWalkNet(params, params.n_classes, params.net_input_dim, model_fn,
                                       model_must_be_load=True, dump_model_visualization=False)


  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  n_walks = 0
  model_fns = pred_all = lbl_all = None
  print('Calculating embeddings.')
  tb = time.time()
  for iter in range(n_iters):
    for name_, model_ftrs, labels in test_dataset:
      labels = labels.numpy()
      name = name_.numpy()[0].decode()
      print('  - Got data', name, labels)
      sp = model_ftrs.shape
      ftrs = tf.reshape(model_ftrs, (-1, sp[-2], sp[-1]))
      #print('  - Start Run Pred')
      predictions_features = dnn_model(ftrs, training=False, classify=False).numpy()
      if 0:
        predictions_features = np.mean(predictions_features, axis=0)[None, :]
        name = [name]
      else:
        labels = np.repeat(labels, predictions_features.shape[0])
        name = [name] * predictions_features.shape[0]
      predictions_labels   = dnn_model(ftrs, training=False, classify=True).numpy()
      #print('  - End Run Pred')
      pred_best = predictions_labels.argmax(axis=1)
      acc = np.mean(labels == pred_best)
      print('This batch accuracy:', round(100 * acc, 2))
      if pred_all is None:
        pred_all = predictions_features
        lbl_all = labels
        model_fns = name
      else:
        pred_all = np.vstack((pred_all, predictions_features))
        lbl_all = np.concatenate((lbl_all, labels))
        model_fns += name
      #if pred_all.shape[0] > 1200:
      #  break
      #break
  print('Feature calc time: ', round(time.time() - tb, 2))
  shape_fn_2_id = {}
  shape_fns = np.array(model_fns)
  for cls in np.unique(lbl_all):
    this_cls_idxs = np.where(lbl_all == cls)[0]
    shape_fn_this_cls = shape_fns[this_cls_idxs]
    shape_fn_2_id[cls] = {n: i for i, n in enumerate(list(set(shape_fn_this_cls)))}
  if 0:
    pred_all = pred_all[:1200, :20]
    lbl_all = lbl_all[:1200]
  print('Embedding shape:', pred_all.shape)
  print('t-SNE calculation')
  transformer = TSNE(n_components=2)
  ftrs_tsne = transformer.fit_transform(pred_all)
  print('  t-SNE calc finished')
  shps = '.1234+X|_'
  shps = '.<*>^vspPDd'
  colors = utils.colors_list
  plt.figure()
  i_cls = -1
  for cls, this_cls_shape_fns in shape_fn_2_id.items():
    i_cls += 1
    for i_shape, this_shape_fn in enumerate(this_cls_shape_fns):
      idxs = (shape_fns == this_shape_fn) * (lbl_all == cls)
      if idxs.size:
        clr = colors[i_cls % len(colors)]
        edgecolor = colors[(i_shape + 1) % len(colors)]
        mrkr = shps[i_shape % len(shps)]
        if i_shape == 0:
          label=dataset_labels[cls]
        else:
          label = None
        plt.scatter(ftrs_tsne[idxs, 0], ftrs_tsne[idxs, 1], color=clr, marker=mrkr, #edgecolor=edgecolor, linewidth=3,
                    s=100, label=label)
  plt.legend(fontsize=15)
  plt.axis('off')
  plt.show()

def check_rotation_weak_points():
  if 0:
    logdir = '/home/alonla/mesh_walker/runs_aug_45/0001-06.08.2020..17.40__shrec11_10-10_A/'
    logdir = '/home/alonla/mesh_walker/runs_aug_45/0002-06.08.2020..21.48__shrec11_10-10_B/'
  else:
    logdir = '/home/alonla/mesh_walker/runs_aug_360/0001-06.08.2020..17.40__shrec11_10-10_A/'
    logdir = '/home/alonla/mesh_walker/runs_aug_360_must/0001-23.08.2020..18.03__shrec11_10-10_A/'
    #logdir = '/home/alonla/mesh_walker/runs_aug_360/0002-06.08.2020..21.49__shrec11_10-10_B/'
  print(logdir)
  dataset_folder = '/home/alonla/mesh_walker/datasets_processed/shrec11/10-10_A/test/*.*'
  #dataset_folder = '/home/alonla/mesh_walker/datasets_processed/shrec11/10-10_B/test/*.*'
  model_fn = logdir + 'learned_model2keep__00060003.keras'
  dnn_model = None
  rot_angles = range(0, 360, 10)
  for axis in [0, 1, 2]:
    accs = []
    stds = []
    dataset.data_augmentation_rotation.test_rotation_axis = axis
    for rot in rot_angles:
      accs_this_rot = []
      for _ in range(5):
        acc, dnn_model = calc_accuracy_test(logdir=logdir, dataset_folder=dataset_folder, n_walks_per_model=16,
                                            dnn_model=dnn_model, labels=dataset_prepare.shrec11_labels,
                                            model_fn=model_fn, verbose_level=0, data_augmentation={'rotation': rot})
        accs_this_rot.append(acc[0])
      accs.append(np.mean(accs_this_rot))
      stds.append(np.std(accs_this_rot))
      print(rot, accs, stds)
    plt.errorbar(rot_angles, accs, yerr=stds)
    plt.xlabel('Rotation [degrees]')
    plt.ylabel('Accuracy')
    plt.title('Accuracy VS rotation angles, axis = ' + str(axis))
  plt.legend(['axis=0', 'axis=1', 'axis=2'])
  plt.suptitle('/'.join(logdir.split('/')[-3:-1]))
  plt.show()


if __name__ == '__main__':
  np.random.seed(0)
  utils.config_gpu(1)

  if 0:
    check_rotation_weak_points()
    exit(0)

  #test_dataset()
  iter2use = 'last'
  classes_indices_to_use = None
  model_fn = None
  if 0:   # t-SNE
    if 1:   # Use shrec model
      r = 'shrec11_2keep/0005-07.05.2020..15.08__Shrec11_16-04A_CyclicLR'
      logdir = os.path.expanduser('~') + '/mesh_walker/mesh_learning/' + r + '/'
      model_fn = logdir + 'learned_model2keep__00060000.keras'
      d = 'shrec11/16-04_A/test/*.npz'
      cls2show = [0, 1, 2, 32, 7, 4, 15] # [0, 1, 2, 3, 4, 6, 7, 9][:5]
      dataset_labels = dataset_prepare.shrec11_labels
    else:   # Use ModelNet
      r = 'rnn_mesh_walk/0012-11.05.2020..19.08__modelnet__'
      d = 'modelnet40_1k2k4k/*test*.npz'
      cls2show = [0, 3, 5, 8, 10, 18, 19, 32, 35][:3]
      dataset_labels = dataset_prepare.model_net_labels
      logdir = os.path.expanduser('~') + '/mesh_walker/mesh_learning/' + r + '/'
      model_fn = logdir + 'learned_model2keep__00100520.keras'
    show_features_tsne(from_cach_dataset=d, logdir=logdir, n_iters=1, cls2show=cls2show, dataset_labels=dataset_labels,
                       model_fn=model_fn, max_size_per_class=3)

  elif 0:
    logdir = '/home/ran/mesh_walker/runs_compare/0095-23.11.2020..15.31__modelnet'
    import glob, os
    model_fn = glob.glob(os.path.join(logdir, 'learned_model2keep__*1802*.keras'))[
      -1]  # logdir + 'learned_model2keep__00100520.keras'
    dataset_folder = os.path.expanduser('~') + '/mesh_walker/datasets/3dFUTURE/*test*/*.npz'
    min_max_faces2use = [000, 4000]
    # classes_indices_to_use = [3, 5, 10]
    accs, _ = calc_accuracy_test(logdir=logdir, dataset_folder=dataset_folder,
                                 labels=dataset_prepare.future3d_labels, iter2use=iter2use,
                                 classes_indices_to_use=classes_indices_to_use,
                                 min_max_faces2use=min_max_faces2use, model_fn=model_fn, n_walks_per_model=16 * 4)
    print('Overall Accuracy / Mean Accuracy:', np.round(np.array(accs) * 100, 2))
  elif 1: # ModelNet
    # logdir = '/home/ran/mesh_walker/runs_compare/0250-27.12.2020..09.20__modelnet_80_20_triplet'
    # logdir = '/home/ran/mesh_walker/runs_compare/0212-12.12.2020..16.23__modelnet_triplet'
    logdir = '/home/ran/mesh_walker/runs_compare/0168-03.12.2020..12.37__modelnet_multiwalk'
    import glob, os
    # model_fn = glob.glob(os.path.join(logdir, 'learned_model2keep__*.keras')) # logdir + 'learned_model2keep__00100520.keras'
    model_fn = glob.glob(os.path.join(logdir,'learned_model2keep__*1202*.keras'))
    model_fn.sort()
    model_fn = model_fn[-1]
    dataset_folder = os.path.expanduser('~') + '/mesh_walker/datasets/modelnet40_reported/*test*.npz'
    # dataset_folder = os.path.expanduser('~') + '/mesh_walker/datasets/modelnet40_retrieval_split_0/*test*.npz'
    min_max_faces2use = [000, 4000]
    # classes_indices_to_use = [3, 5, 10]
    accs, _ = calc_accuracy_test(logdir=logdir, dataset_folder=dataset_folder,
                       labels=dataset_prepare.model_net_labels, iter2use=iter2use, classes_indices_to_use=classes_indices_to_use,
                       min_max_faces2use=min_max_faces2use, model_fn=model_fn, n_walks_per_model=32)
    for k,v in accs.items():
      print('{}:{:2.3f}'.format(k, v))
  elif 1:
    if 0:
      ra = '0069-08.02.2020..21.02__Shrec16_XyzAndJumps_NoLocalJumpWalk_AugmentationsAdded_Split10-10A'
      rb = '0070-08.02.2020..21.03__Shrec16_XyzAndJumps_NoLocalJumpWalk_AugmentationsAdded_Split10-10B'
      rc = '0071-08.02.2020..21.03__Shrec16_XyzAndJumps_NoLocalJumpWalk_AugmentationsAdded_Split10-10C'
      dataset_path_a = 'shrec11/10-10_A/test/*.npz'
      dataset_path_b = 'shrec11/10-10_B/test/*.npz'
      dataset_path_c = 'shrec11/10-10_C/test/*.npz'
    else:
      ra = 'runs_compare/0001-12.11.2020..14.30__shrec11_10-10_A'
      rb = 'runs_compare/0005-12.11.2020..14.39__shrec11_10-10_B'
      rc = 'runs_compare/0006-12.11.2020..14.39__shrec11_10-10_C'
      dataset_path_a = os.path.expanduser('~') + '/mesh_walker/datasets/shrec11/10-10_A/test/*.npz'
      dataset_path_b = os.path.expanduser('~') + '/mesh_walker/datasets/shrec11/10-10_B/test/*.npz'
      dataset_path_c = os.path.expanduser('~') + '/mesh_walker/datasets/shrec11/10-10_C/test/*.npz'
    acc_all = {}
    times = []
    for r, dataset_path in zip([ra, rb, rc], [dataset_path_a, dataset_path_b, dataset_path_c]):
      s_time = time.time()
      for k in [1, 2, 4]:
        acc, _ = calc_accuracy_test(logdir=os.path.expanduser('~') + '/mesh_walker/' + r + '/',
                                    dataset_folder=dataset_path, labels=dataset_prepare.shrec11_labels, iter2use=iter2use,
                                    n_walks_per_model=k, verbose_level=1)
        acc_all.setdefault(k, []).append(acc['overall_accuracy'])
      times.append(time.time() - s_time)

      # break
    for k,v in acc_all.items():
      print(k, np.mean(v))
    print(times)
    print('{:2.5f}'.format(np.mean(times)))
    print(acc_all)
    print('debug')
    # print(np.mean(acc_all))
  elif 1: # Look for Rotation weekpoints
    if 0:
      logdir = '/home/alonla/mesh_walker/runs_aug_360/0004-07.08.2020..06.05__shrec11_16-04_A'
      dataset_folder = '/home/alonla/mesh_walker/datasets_processed/shrec11/16-04_A/test/*.*'
    else:
      if 0:
        logdir = '/home/alonla/mesh_walker/runs_aug_45/0001-06.08.2020..17.40__shrec11_10-10_A/'
      else:
        logdir = '/home/alonla/mesh_walker/runs_aug_360/0001-06.08.2020..17.40__shrec11_10-10_A/'
      dataset_folder = '/home/alonla/mesh_walker/datasets_processed/shrec11/10-10_A/test/*.*'
    model_fn = None # logdir + 'learned_model2keep__00200010.keras'
    tb = time.time()
    dnn_model = None
    accs = []
    rot_angles = range(0, 180, 10)
    for rot in rot_angles:
      acc, dnn_model = calc_accuracy_test(logdir=logdir, dataset_folder=dataset_folder, n_walks_per_model=8,
                                          dnn_model=dnn_model, labels=dataset_prepare.shrec11_labels, iter2use=str(iter2use),
                                          model_fn=model_fn, verbose_level=0, data_augmentation={'rotation': rot})
      accs.append(acc[0])
      print(rot, accs)
    plt.plot(rot_angles, accs)
    plt.xlabel('Rotation [degrees]')
    plt.ylabel('Accuracy')
    plt.title('Accuracy VS rotation angles')
    plt.show()
  elif 1: # Check STD vs Number of walks
    if 1:
      logdir = '/home/alonla/mesh_walker/runs_aug_360/0004-07.08.2020..06.05__shrec11_16-04_A'
      dataset_folder = '/home/alonla/mesh_walker/datasets_processed/shrec11/16-04_A/test/*.*'
    else:
      logdir = '/home/alonla/mesh_walker/runs_aug_360/0001-06.08.2020..17.40__shrec11_10-10_A'
      dataset_folder = '/home/alonla/mesh_walker/datasets_processed/shrec11/10-10_A/test/*.*'
    model_fn = None # logdir + 'learned_model2keep__00200010.keras'
    tb = time.time()
    dnn_model = None
    for n_walks in [1, 2, 4, 8, 16, 32]:
      accs = []
      for _ in range(6):
        acc, dnn_model = calc_accuracy_test(logdir=logdir, dataset_folder=dataset_folder, n_walks_per_model=n_walks,
                                            dnn_model=dnn_model, labels=dataset_prepare.shrec11_labels, iter2use=str(iter2use),
                                            model_fn=model_fn, verbose_level=0)
        accs.append(acc[0])
        #print('Run Time: ', time.time() - tb, ' ; Accuracy:', acc)
      print(n_walks, accs, 'STD:', np.std(accs))
  elif 1:
    r = 'cubes2keep/0016-03.04.2020..08.59__Cubes_NewPrms'
    logdir = os.path.expanduser('~') + '/mesh_walker/mesh_learning/' + r + '/'
    model_fn = logdir + 'learned_model2keep__00160080.keras'
    n_walks_to_check = [64, 1, 2, 4, 8, 16, 32]
    acc_all = []
    for n_walks_per_model in n_walks_to_check:
      acc = calc_accuracy_test(logdir=logdir, model_fn=model_fn, target_n_faces=[1000],
                         from_cach_dataset='cubes/test*.npz', labels=dataset_prepare.cubes_labels, n_walks_per_model=n_walks_per_model)
      acc_all.append(acc[0][0])
    print('--------------------------------')
    print(acc_all)
    #[0.7708649468892261, 0.8482549317147192, 0.921092564491654, 0.952959028831563, 0.9742033383915023, 0.9787556904400607]
  #calc_accuracy_per_seq_len()
  #calc_accuracy_per_n_faces()
  #features_analysis()
