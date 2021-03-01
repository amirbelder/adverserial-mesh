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

timelog = {}
timelog['prep_model'] = []
timelog['fill_features'] = []


def Eu_dis_mat_fast(X):
  aa = np.sum(np.multiply(X, X), 1)
  ab = X * X.T
  D = aa + aa.T - 2 * ab
  D[D < 0] = 0
  D = np.sqrt(D)
  D = np.maximum(D, D.T)
  return D

def calculate_map(fts, lbls, dis_mat=None, top_k=1000):
  if dis_mat is None:
    dis_mat = Eu_dis_mat_fast(np.mat(fts))
  num = len(lbls)
  mAP = 0
  for i in range(num):
    scores = dis_mat[:, i]
    targets = (lbls == lbls[i]).astype(np.uint8)
    sortind = np.argsort(scores, 0)[:top_k]
    truth = targets[sortind]
    sum = 0
    precision = []
    for j in range(top_k):
      if truth[j]:
        sum += 1
        precision.append(sum * 1.0 / (j + 1))
    if len(precision) == 0:
      ap = 0
    else:
      for ii in range(len(precision)):
        precision[ii] = max(precision[ii:])
      ap = np.array(precision).mean()
    mAP += ap
    # print(f'{i+1}/{num}\tap:{ap:.3f}\t')
  mAP = mAP / num
  return mAP


def cal_pr(cfg, des_mat, lbls, save=True, draw=False, top_k=1000):
    num = len(lbls)
    precisions = []
    recalls = []
    ans = []
    for i in range(num):
        scores = des_mat[:, i]
        targets = (lbls == lbls[i]).astype(np.uint8)
        sortind = np.argsort(scores, 0)[:top_k]
        truth = targets[sortind]
        tmp = 0
        sum = truth[:top_k].sum()
        precision = []
        recall = []
        for j in range(top_k):
            if truth[j]:
                tmp+=1
                # precision.append(sum/(j + 1))
            recall.append(tmp*1.0/sum)
            precision.append(tmp*1.0/(j+1))
        precisions.append(precision)
        for j in range(len(precision)):
            precision[j] = max(precision[j:])
        recalls.append(recall)
        tmp = []
        for ii in range(11):
            min_des = 100
            val = 0
            for j in range(top_k):
                if abs(recall[j] - ii * 0.1) < min_des:
                    min_des = abs(recall[j] - ii * 0.1)
                    val = precision[j]
            tmp.append(val)
        print('%d/%d'%(i+1, num))
        ans.append(tmp)
    ans = np.array(ans).mean(0)
    if save:
        save_dir = os.path.join(cfg.result_sub_folder, 'pr.csv')
        np.savetxt(save_dir, np.array(ans), fmt='%.3f', delimiter=',')
    if draw:
        plt.plot(ans)
        plt.show()



def get_model_names():
  part = 'test'
  model_fns = []
  for i, name in enumerate(dataset_prepare.model_net_labels):
    pathname_expansion = os.path.expanduser('~') + '/datasets/ModelNet40/' + name + '/' + part + '/*.off'
    filenames = glob.glob(pathname_expansion)
    model_fns += filenames
  return model_fns

def show_walk(model, features, one_walk=False):
  for wi in range(features.shape[0]):
    walk = features[wi, :, -1].astype(np.int)
    jumps = features[wi, :, -2].astype(np.bool)
    utils.visualize_model_walk(model['vertices'], model['faces'], walk, jumps)
    if one_walk:
      break


def calc_retrieval_test(dataset_folder=False, logdir=None, labels=None, iter2use='last', classes_indices_to_use=None,
                       dnn_model=None, params=None, verbose_level=2, min_max_faces2use=[0, 4000], model_fn=None,
                       target_n_faces=['according_to_dataset'], n_walks_per_model=16, seq_len=None, data_augmentation={}):
  SHOW_WALK = 0
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
  if 0:
    params.net_input.append('jump_indication')

  params.classes_indices_to_use = None
  if seq_len:
    params.seq_len = seq_len
  if verbose_level:
    print('params.seq_len:', params.seq_len, ' ; n_walks_per_model:', n_walks_per_model)

  if SHOW_WALK:
    params.net_input += ['vertex_indices']

  params.set_seq_len_by_n_faces = 1
  if dataset_folder:
    size_limit = np.inf # 200
    params.classes_indices_to_use = classes_indices_to_use
    pathname_expansion = dataset_folder
    if 1:
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
      dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim - SHOW_WALK, model_fn,
                                       model_must_be_load=True, dump_model_visualization=False)

  n_pos_all = 0
  n_classes = 40
  all_confusion = np.zeros((n_classes, n_classes), dtype=np.int)
  size_accuracy = []
  ii = 0
  tb_all = time.time()
  res_per_n_faces = {}
  pred_per_model_name = {}
  pred_per_model_per_faces = {}
  dnn_inference_time = [] # 150mSec for 64 walks of 200 steps
  bad_pred = EasyDict({'n_comp': [], 'biggest_comp_area_ratio': []})
  good_pred = EasyDict({'n_comp': [], 'biggest_comp_area_ratio': []})
  all_features = []
  all_labels = []
  for i, data in tqdm(enumerate(test_dataset), disable=print_details, total=n_models_to_test):
    name, ftrs, gt = data
    model_fn = name.numpy()[0].decode()
    model_name, n_faces = utils.get_model_name_from_npz_fn(model_fn)
    assert ftrs.shape[0] == 1, 'Must have one model per batch for test'
    ftrs = tf.reshape(ftrs, ftrs.shape[1:])
    gt = gt.numpy()[0]
    predictions = None
    features = None
    for i_f, this_target_n_faces in enumerate(target_n_faces):
      model = None
      if SHOW_WALK:
        if model is None:
          model = dataset.load_model_from_npz(model_fn)
        ftrs = ftrs[:, :, :-1]
      ftr2use = ftrs.numpy()
      tb = time.time()
      predictions_, features_ = [x.numpy() for x in dnn_model(ftr2use, classify=both, training=False)]
      te = time.time() - tb
      dnn_inference_time.append(te / n_walks_per_model * 1000)
      if features is None:
        features = features_
      else:
        features = np.vstack((features, features_))


    # TODO: change to retrieval masking - aggregate all model features, retrieval by list?
    all_features.append(features)
    all_labels.append(gt)

    mean_pred = np.mean(predictions, axis=0)
    max_hit = np.argmax(mean_pred)

    if model_name not in pred_per_model_name.keys():
      pred_per_model_name[model_name] = [gt, np.zeros_like(mean_pred)]
    pred_per_model_name[model_name][1] += mean_pred

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

  if print_details:
    print(utils.color.BLUE + 'Total time, all:', time.time() - tb_all, utils.color.END)

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
    if 'future' in params.logdir:
      labels_file = '/home/ran/Databases/3D-FUTURE/GT/model_infos.json'
      with open(labels_file, 'r') as f:
        labels_per_id = {x['model_id']: x['category'] for x in json.load(f) if not x['is_train']}
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
        m_name = labels_per_id[m_name.split('_')[-1]]
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
    fn = 'False_preds.csv'
    with open(os.path.join(params.logdir, fn), 'w') as f:
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
        with open(os.path.join(params.logdir, 'acc_per_class.txt'), 'wa') as f:
          f.writeline(msg)
        print(str(i).ljust(3), name.ljust(12), n_this_type, ',', str(round(accuracy_this_type * 100, 1)).ljust(5), ' ; 2nd best:', scnd_best_name.ljust(12), round(accuracy_2nd_best * 100, 1))
  mean_acc_per_class = np.mean(acc_per_class)

  if 0:
    print('Time Log:')
    for k, v in timelog.items():
      print('  ' , k, ':', np.mean(v))

  return {'overall_accuracy': mean_accuracy_all_faces,
          'class_mean_accuracy': mean_acc_per_class
          }, dnn_model



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

  elif 1:
    logdir = '/home/ran/mesh_walker/runs_compare/0099-24.11.2020..10.44__future_attention_400'
    import glob, os
    model_fn = glob.glob(os.path.join(logdir, 'learned_model2keep__*60358.keras'))[
      -1]  # logdir + 'learned_model2keep__00100520.keras'
    dataset_folder = os.path.expanduser('~') + '/mesh_walker/datasets/3dFUTURE/*test*/*.npz'
    min_max_faces2use = [000, 4000]
    # classes_indices_to_use = [3, 5, 10]
    accs, _ = calc_accuracy_test(logdir=logdir, dataset_folder=dataset_folder,
                                 labels=dataset_prepare.future3d_labels, iter2use=iter2use,
                                 classes_indices_to_use=classes_indices_to_use,
                                 min_max_faces2use=min_max_faces2use, model_fn=model_fn, n_walks_per_model=16 * 4)
    print('Overall Accuracy / Mean Accuracy:', np.round(np.array(accs) * 100, 2))
  elif 0: # ModelNet
    logdir = '/home/ran/mesh_walker/runs_aug_360_must/0033-11.10.2020..06.28__modelnet_attention'
    import glob, os
    model_fn = glob.glob(os.path.join(logdir, 'learned_model2keep__*.keras'))[-1] # logdir + 'learned_model2keep__00100520.keras'
    dataset_folder = os.path.expanduser('~') + '/mesh_walker/datasets/modelnet40_reported/*test*.npz'
    min_max_faces2use = [000, 4000]
    # classes_indices_to_use = [3, 5, 10]
    accs, _ = calc_accuracy_test(logdir=logdir, dataset_folder=dataset_folder,
                       labels=dataset_prepare.model_net_labels, iter2use=iter2use, classes_indices_to_use=classes_indices_to_use,
                       min_max_faces2use=min_max_faces2use, model_fn=model_fn, n_walks_per_model=16 * 4)
    print('Overall Accuracy / Mean Accuracy:', np.round(np.array(accs) * 100, 2))
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
    n_walks_to_check = [1, 2, 4, 8, 16, 32]
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
