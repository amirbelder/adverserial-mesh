import os

from easydict import EasyDict
import numpy as np

import utils
import dataset_prepare
import dataset
from scannet_preprocess.dataset_prepare_scannet_v2 import scannet_weights
from dataset_prepare import future3d_weights

if 0:
  MAX_AUGMENTATION = 90
  run_folder = 'runs_test'
elif 0:
  MAX_AUGMENTATION = 45
  run_folder = 'runs_aug_45'
else:
  MAX_AUGMENTATION = 360

run_folder = 'runs_compare'

def multiwalk_params(params, n_walks=8, train_head_only=False, pretrained_weights_path=None):
  params.cross_walk_attn = True
  params.train_multiwalk_head = False
  params.n_walks_per_model = n_walks
  params.batch_size = int(64 / params.n_walks_per_model)
  params.full_accuracy_test['n_walks_per_model'] = params.n_walks_per_model
  params.logdir += '_multiwalk'

  # Retrain params
  if train_head_only:
    params.train_multiwalk_head = True
    params.optimizer_type = 'cycle'  # 'transformer'
    params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-5,
                                      'maximal_learning_rate': 1e-4,
                                      'step_size': 2000})
    import glob
    paths = glob.glob(pretrained_weights_path)
    paths.sort()
    params.net_start_from_prev_net = paths[-1]
    if params.triplet_loss:
      params.centers_weights = glob.glob('/'.join(pretrained_weights_path.split('/')[:-1] + ['TCL_centers*.keras']))
      params.centers_weights.sort()
      params.centers_weights = params.centers_weights[-1]
  else:
    params.logdir += '_scratch'
  return params


def set_up_default_params(network_task, run_name, cont_run_number=0):
  '''
  Define dafault parameters, commonly for many test case
  '''
  params = EasyDict()

  params.cont_run_number = cont_run_number
  params.run_root_path = os.path.expanduser('~') + '/mesh_walker/' + run_folder
  params.logdir = utils.get_run_folder(params.run_root_path + '/', '__' + run_name, params.cont_run_number)
  params.model_fn = params.logdir + '/learned_model.keras'

  params.datasets_folder = os.path.expanduser('~') + '/mesh_walker/datasets/'
  # Optimizer params
  params.optimizer_type = 'cycle'  # sgd / adam / cycle
  params.learning_rate = [5e-5]
  params.learning_rate_dynamics = 'cycle'
  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6,
                                    'maximal_learning_rate': 5e-4,
                                    'step_size': 10000})
  params.n_models_per_test_epoch = 300
  params.gradient_clip_th = 1

  # Dataset params
  params.classes_indices_to_use = None
  params.train_dataset_size_limit = np.inf
  params.test_dataset_size_limit = np.inf
  params.network_task = network_task
  params.normalize_model = True
  params.sub_mean_for_data_augmentation = True
  params.datasets2use = {}
  params.test_data_augmentation = {}
  params.train_data_augmentation = {'aspect_ratio': 0.3,
                                    'translation': 0.2,
                                    'jitter': 0.05,
                                    }
  params.aditional_network_params = []  #['pooling']
  params.cut_walk_at_deadend = False

  params.network_tasks = [params.network_task]
  if params.network_task == 'classification':
    params.n_walks_per_model = 1
    params.one_label_per_model = True
    params.train_loss = ['cros_entr']
  elif params.network_task == 'semantic_segmentation':
    params.n_walks_per_model = 4
    params.one_label_per_model = False
    params.train_loss = ['cros_entr']
  else:
    raise Exception('Unsuported params.network_task: ' + params.network_task)
  params.batch_size = int(32 / params.n_walks_per_model)

  # Other params
  params.log_freq = 100
  params.walk_alg = 'random_global_jumps'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  params.net_input = ['xyz'] # 'xyz', 'dxdydz', 'jump_indication'
  params.reverse_walk = False
  params.train_min_max_faces2use = [0, np.inf]
  params.test_min_max_faces2use = [0, np.inf]

  params.net = 'RnnWalkNet'
  params.last_layer_actication = 'softmax'
  params.use_norm_layer = 'InstanceNorm'  # BatchNorm / InstanceNorm / None
  params.layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512}
  params.initializers = 'orthogonal'
  params.adjust_vertical_model = False
  params.net_start_from_prev_net = None

  params.net_gru_dropout = 0
  params.uniform_starting_point = False
  params.train_max_size_per_class = None    # None / 'uniform_as_max_class' / <a number>

  params.full_accuracy_test = None

  params.iters_to_train = 100e3
  params.triplet_loss = False
  params.cross_walk_attn = False
  params.train_multiwalk_head = False
  return params

# Classifications
# ---------------
def modelnet_params(mode='rnnWalk', short=False, N_per_class=False):
  params = set_up_default_params('classification', 'modelnet', 0)
  params.n_classes = 40
  if short:
    p = 'modelnet40_80_20'
    params.datasets2use['train'] = [os.path.expanduser('~') + '/mesh_walker/datasets/' + p + '/*train*.npz']
    params.datasets2use['test'] = [os.path.expanduser('~') + '/mesh_walker/datasets/' + p + '/*test*.npz']
    # == Running on 80-20 split for retrieval (same as
    params.retrieval_80_20 = True
    # params.logdir += '_80_20'
    # # params.optimizer_type = 'adam'
    # # params.learning_rate = [1e-4]
    params.triplet_loss = True
    params.triplet_dim = params.n_classes # params.layer_sizes['gru3']  #params.n_classes #  512
    params.centers_weights = None
    params.logdir += '_triplet'

    # add rotations every 30 degrees to compare correctly, for train we can augment on-the-fly,
    # for test we must create augmentations as multiple models
    # params.train_data_augmentation = {'const_rotation': 30}
  else:
    p = 'modelnet40_reported'
    # p = 'modelnet40_upsample_new'
    params.datasets2use['train'] = [os.path.expanduser('~') + '/mesh_walker/datasets/' + p + '/*train*.npz']
    params.datasets2use['test'] = [os.path.expanduser('~') + '/mesh_walker/datasets/' + p + '/*test*.npz']
    # params.class_weights = dataset_prepare.model_net_weights
    # params.logdir += '_CBL'   # class balanced loss

  params.train_min_max_faces2use = [0000, 4000]
  params.test_min_max_faces2use = [0000, 4000]

  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6,
                                    'maximal_learning_rate': 0.0005,
                                    'step_size': 5000})



  params.seq_len = 800
  params.min_seq_len = int(params.seq_len / 2)

  params.full_accuracy_test = {'dataset_folder': params.datasets2use['test'][0],
                               'labels': dataset_prepare.model_net_labels,
                               'min_max_faces2use': params.test_min_max_faces2use,
                               'n_walks_per_model': 16 * 4,
                               }


  # Parameters to recheck:
  params.iters_to_train = 500e3
  params.net_input = ['xyz']  #, 'v_normals']
  # params.net_input = ['xyz']
  params.walk_alg = 'random_global_jumps'   # no_jumps / global_jumps

  # # === TCL params ==== #
  # params.triplet_loss = True
  # params.triplet_dim = params.layer_sizes['gru3']  #params.n_classes #  512
  # params.centers_weights = None
  # params.logdir += '_triplet'

  # =================== #
  params.cross_walk_attn = False
  params.in_walk_attn = False

  # Attention params for new network:
  if mode == 'Attention':
    params.optimizer_type = 'cycle'  # 'transformer'
    params.cycle_opt_prms = EasyDict({'initial_learning_rate': 5e-7,
                                      'maximal_learning_rate': 1e-4,
                                      'step_size': 10000})
    params.net = 'HierTransformer'
    # params.walk_alg = 'constant_global_jumps_new'  # 'constant_global_jumps_bfs'   #'constant_global_jumps'
    params.walk_alg = 'random_global_jumps'   #'constant_global_jumps'
    params.net_params = EasyDict({'num_layers': 3,  # number of stacked layers in transformer
                                  'num_scales': 1,
                                  'd_model': 512,  # dimension of feature vector
                                  'num_heads': 1,  # number of parallel multi-heads
                                  'dff': 2048,  # dimension of feed-forward at the end of each encoder\decoder unit
                                  'input_vocab_size': 20,  # check if relevant should be deleted later
                                  'out_features': 512,
                                  'jump_every_k': 20,
                                  'pe_input': 20,  # maximum positional encoding for input - ?! check effect
                                  'pe_target': 20,  # maximum positional encoding for target - ?! check effect
                                  'global_dim_mult': 1,
                                  'rate': 0.0,
                                  'pooling': True,
                                  'concat_xyz': False,
                                  })
    params.logdir += '_attention'

    # import glob
    # params.net_start_from_prev_net = \
    # glob.glob('/home/ran/mesh_walker/runs_compare/0174-07.12.2020..15.38__modelnet_attention/*.keras')[-1]
    # # If we want cross-walk attention
    # params.cross_walk_attn = True
    # params.n_walks_per_model = 4
    # params.batch_size = int(64 / params.n_walks_per_model)
    # params.full_accuracy_test['n_walks_per_model'] = params.n_walks_per_model
    # params.logdir += '_multiwalk'

  elif mode == 'rnnAttention':
    params.net = 'rnnAttention'
    params.logdir += '_rnnAttention'

  elif mode == 'Multiwalk':
    if short:
      weights_path = '/home/ran/mesh_walker/runs_compare/0308-12.01.2021..15.22__modelnet_80_20_triplet_400/learned_model2keep__*.keras'
    elif params.triplet_loss:
      weights_path = '/home/ran/mesh_walker/runs_compare/0229-20.12.2020..11.54__modelnet_triplet/learned_model2keep__*.keras'
    else:
      # weights_path = '/home/ran/mesh_walker/runs_compare/0128-29.11.2020..15.59__modelnet_multiwalk/learned_model2keep__*.keras'
      weights_path = '/home/ran/mesh_walker/runs_best/0095-23.11.2020..15.31__modelnet/learned_model2keep__*180*.keras'
    params = multiwalk_params(params, n_walks=8, train_head_only=True, pretrained_weights_path=weights_path)
    # params = multiwalk_params(params, n_walks=16)  #, train_head_only=True, pretrained_weights_path=weights_path)

    # params = multiwalk_params(params, n_walks=8)

  params.logdir += '_{}'.format(mode)
  params.logdir += '_' + '_'.join(p.split('_')[1:])
  params.logdir += '_{}'.format(params.seq_len)
  #
  # params.multiscale = True
  # params.n_walks_per_model = 4
  # params.batch_size = 16
  # params.logdir += '_ms'
  return params


def scannet_params(mode='rnnWalk'):
  # third axis is Z
  params = set_up_default_params('semantic_segmentation', 'scannet_v2', 0)
  params.n_classes = 20  # remember we exclude label 0 - need to +1 all label predictions!
  params.class_weights = scannet_weights


  params.normalize_model = False
  params.center_model_s3dis = False
  params.sub_mean_for_data_augmentation = False

  params.ignore_label = -1

  params.batch_size = int(8 / params.n_walks_per_model)
  p = 'scannet_v2_4cm_crops'
  params.train_min_max_faces2use = [0000, np.inf]
  params.test_min_max_faces2use = [0000, np.inf]

  params.datasets2use['train'] = [params.datasets_folder + p + '/*train*.npz']
  params.datasets2use['test'] = [params.datasets_folder + p + '/*val*.npz']

  params.seq_len = 400
  params.min_seq_len = int(params.seq_len / 2)

  params.full_accuracy_test = {'dataset_folder': params.datasets2use['test'][0],
                               'n_iters': 4,
                               }


  # Parameters to recheck:
  params.iters_to_train = 300e3
  params.net_input = ['dxdydz', 'rgb', 'v_normals']
  params.walk_alg = 'no_jumps'   # no_jumps / global_jumps / random_global_jumps
  # params.train_data_augmentation = {'rotation': MAX_AUGMENTATION}


  # Attention params for new network:
  if mode == 'Attention':
    params.optimizer_type = 'cycle'  # 'transformer'
    params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6,
                                      'maximal_learning_rate': 1e-4,
                                      'step_size': 10000})
    params.net = 'HierTransformer'
    # params.walk_alg = 'constant_global_jumps_new'  # 'constant_global_jumps_bfs'   #'constant_global_jumps'
    params.walk_alg = 'euclidean_jumps'   #'constant_global_jumps'
    params.net_params = EasyDict({'num_layers': 3,  # number of stacked layers in transformer
                                  'num_scales': 1,
                                  'd_model': 512,  # dimension of feature vector
                                  'num_heads': 8,  # number of parallel multi-heads
                                  'dff': 2048,  # dimension of feed-forward at the end of each encoder\decoder unit
                                  'input_vocab_size': 20,  # check if relevant should be deleted later
                                  'out_features': 512,
                                  'jump_every_k': 20,
                                  'pe_input': 20,  # maximum positional encoding for input - ?! check effect
                                  'pe_target': 20,  # maximum positional encoding for target - ?! check effect
                                  'global_dim_mult': 1,
                                  'rate': 0.0,
                                  'pooling': False,
                                  'concat_xyz': False,
                                  })
    params.logdir += '_attention'
  #params.train_max_size_per_class = 20
  elif mode == 'rnnAttention':
    params.net = 'rnnAttention'
    params.logdir += '_rnnAttention'
  return params


def cubes_params(mode='rnn'):
  # |V| = 250 , |F| = 500 => seq_len = |V| / 2.5 = 100
  params = set_up_default_params('classification', 'cubes', 0)
  params.n_classes = 22
  params.seq_len = 400
  params.min_seq_len = int(params.seq_len / 2)

  p = 'cubes'
  params.datasets2use['train'] = [params.datasets_folder + p + '/*train*.npz']
  params.datasets2use['test'] = [params.datasets_folder + p + '/*test*.npz']

  params.full_accuracy_test = {'dataset_folder': params.datasets2use['test'][0],
                               'labels': dataset_prepare.cubes_labels,
                               }

  params.iters_to_train = 460e3

  # params.optimizer_type = 'adam'  # 'transformer'
  if mode == 'Attention':
    # params.optimizer_type = 'Adam'
    params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6,
                                      'maximal_learning_rate': 5e-5,
                                      'step_size': 10000})
    params.net = 'HierTransformer'
    params.walk_alg = 'random_global_jumps'  # 'constant_global_jumps'
    params.net_params = EasyDict({'num_layers': 3,  # number of stacked layers in transformer
                                  'num_scales': 1,
                                  'd_model': 512,  # dimension of feature vector
                                  'num_heads': 8,  # number of parallel multi-heads
                                  'dff': 2048,  # dimension of feed-forward at the end of each encoder\decoder unit
                                  'input_vocab_size': 20,  # check if relevant should be deleted later
                                  'out_features': 512,
                                  'jump_every_k': 20,
                                  'pe_input': 20,  # maximum positional encoding for input - ?! check effect
                                  'pe_target': 20,  # maximum positional encoding for target - ?! check effect
                                  'global_dim_mult': 1,
                                  'rate': 0.0,
                                  'pooling': True,
                                  'concat_xyz': False,
                                  })
    params.logdir += '_attention'
  elif mode == 'Multiwalk':
    weights_path = '/home/ran/mesh_walker/runs_compare/0179-08.12.2020..11.52__cubes/learned_model2keep__*.keras'
    params = multiwalk_params(params, n_walks=8, train_head_only=True, pretrained_weights_path=weights_path)  #, train_head_only=True, weights_path=weights_path)
  return params


def shrec11_params(split_part, mode='rnn'):
  # split_part is one of the following:
  # 10-10_A / 10-10_B / 10-10_C
  # 16-04_A / 16-04_B / 16-04_C

  # |V| = 250 , |F| = 500 => seq_len = |V| / 2.5 = 100
  params = set_up_default_params('classification', 'shrec11_' + split_part, 0)
  params.n_classes = 30
  params.seq_len = 200
  params.min_seq_len = int(params.seq_len / 2)

  params.datasets2use['train'] = [params.datasets_folder + 'shrec11/' + split_part + '/train/*.npz']
  params.datasets2use['test']  = [params.datasets_folder + 'shrec11/' + split_part + '/test/*.npz']

  params.train_data_augmentation = {'rotation': MAX_AUGMENTATION}

  params.full_accuracy_test = {'dataset_folder': params.datasets2use['test'][0],
                               'labels': dataset_prepare.shrec11_labels,
                               'n_walks_per_model': params.n_walks_per_model}

  # params.cross_walk_attn = True
  # params.logdir += '_cross_walk_attn'

  # if params.train_max_size_per_class:
  #   params.logdir += '_{}_train_per_class'.format(params.train_max_size_per_class)
  if mode == 'Multiwalk':
    weights_path = '/home/ran/mesh_walker/runs_compare/0006-12.11.2020..14.39__shrec11_10-10_C/learned_model2keep__*.keras'
    params = multiwalk_params(params, n_walks=8, train_head_only=True, pretrained_weights_path=weights_path)  #, train_head_only=True, weights_path=weights_path)
    # params = multiwalk_params(params, n_walks=8)  #, train_head_only=True, weights_path=weights_path)

  return params


# Semantic Segmentation
# ---------------------
def human_seg_params(mode='rnn'):
  # |V| = 750 , |F| = 1500 => seq_len = |V| / 2.5 = 300
  params = set_up_default_params('semantic_segmentation', 'human_seg', 0)
  params.n_classes = 9
  params.seq_len = 300
  sub_dir = 'sig17_seg_benchmark-no_simplification'

  if 1: # MeshCNN data
    sub_dir = 'human_seg_from_mcnn_with_nrmls'
  if 0: # Simplification to 1.5k faces
    sub_dir = 'sig17_seg_benchmark-1.5k'
  if 0: # Simplification to 4k faces 4000 / 2 / 2.5 = 800
    sub_dir = 'sig17_seg_benchmark-4k'
    params.seq_len = 1200
  if 0: # Simplification to 6k faces 6000 / 2 / 2.5 = 1200
    sub_dir = 'sig17_seg_benchmark-6k'
    params.seq_len = 2000
  if 0: # Simplification to 8k faces
    sub_dir = 'sig17_seg_benchmark-8k'
    params.seq_len = 1600
    params.batch_size = int(16 / params.n_walks_per_model)
  if 0:
    params.n_target_vrt_to_norm_walk = 3000
    sub_dir = 'sig17_seg_benchmark'
    params.seq_len = 2000
  p = params.datasets_folder + sub_dir + '/'
  params.datasets2use['train'] = [p + '*train*.npz']
  params.datasets2use['test']  = [p + '*test*.npz']

  params.min_seq_len = int(params.seq_len / 2)
  params.train_data_augmentation = {'rotation': MAX_AUGMENTATION}

  params.full_accuracy_test = {'dataset_folder': params.datasets2use['test'][0],
                               'n_iters': 32}

  # Parameters to recheck:
  params.iters_to_train = 20000e3

  #params.net_start_from_prev_net = '/home/alon/mesh_walker/runs_test/0011-12.08.2020..07.32__human_seg-6k_unit_bal_norm_lowerLr/learned_model2keep__00010010.keras'

  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-7,
                                    'maximal_learning_rate': 2e-4,
                                    'step_size': 10000})

  if mode == 'Attention':
    params.optimizer_type = 'cycle'  # 'transformer'
    params.learning_rate = [1e-4]
    params.net = 'HierTransformer'
    params.walk_alg = 'random_global_jumps'  # 'constant_global_jumps'
    params.net_params = EasyDict({'num_layers': 2,  # number of stacked layers in transformer
                                  'num_scales': 1,
                                  'd_model': 512,  # dimension of feature vector
                                  'num_heads': 8,  # number of parallel multi-heads
                                  'dff': 2048,  # dimension of feed-forward at the end of each encoder\decoder unit
                                  'input_vocab_size': None,  # check if relevant should be deleted later
                                  'out_features': 512,
                                  'jump_every_k': params.seq_len,
                                  'pe_input': None,  # maximum positional encoding for input - ?! check effect
                                  'pe_target': None,  # maximum positional encoding for target - ?! check effect
                                  'global_dim_mult': 1,
                                  'rate': 0.1,
                                  'pooling': False,
                                  'concat_xyz': False,
                                  })
    params.logdir += '_attention'

  elif mode == 'Multiwalk':
    weights_path = '/home/ran/mesh_walker/runs_compare/0064-18.11.2020..12.23__human_seg/learned_model2keep*.keras'
    params = multiwalk_params(params, train_head_only=True, pretrained_weights_path=weights_path)



  return params


def coseg_params(type): # aliens / chairs / vases
  # |V| = 750 , |F| = 1500 => seq_len = |V| / 2.5 = 300
  sub_folder = 'coseg_' + type
  p = 'coseg/' + sub_folder + '/'
  params = set_up_default_params('semantic_segmentation', 'coseg_' + type, 0)
  params.n_classes = 10
  params.seq_len = 300
  params.min_seq_len = int(params.seq_len / 2)

  params.datasets2use['train'] = [params.datasets_folder + p + '*train*.npz']
  params.datasets2use['test']  = [params.datasets_folder + p + '*test*.npz']
  params.walk_alg = 'no_jumps'
  params.train_data_augmentation = {'rotation': MAX_AUGMENTATION}

  params.full_accuracy_test = {'dataset_folder': params.datasets2use['test'][0],
                               'n_iters': 32}


  return params


def future_params(mode='rnn'):
  params = set_up_default_params('classification', 'future_raw', 0)
  params.n_classes = 34

  p = '3dFUTURE_raw'
  params.class_weights = future3d_weights
  # TODO: exclude classes with < 30 shapes (Chaise Lounge Sofa, Dressing chair) - not reported in dataset paper
  params.classes_indices_to_use = [x for x in range(params.n_classes) if x not in [19, 30]]   #['Dressing Chair', 'Chaise Longue Sofa']

  params.normalize_model = False
  params.train_min_max_faces2use = [0000, 4000]
  params.test_min_max_faces2use = [0000, 4000]

  params.datasets2use['train'] = [os.path.expanduser('~') + '/mesh_walker/datasets/' + p + '/train*/*.npz']
  params.datasets2use['test'] = [os.path.expanduser('~') + '/mesh_walker/datasets/' + p + '/test*/*.npz']

  params.seq_len = 800
  params.min_seq_len = int(params.seq_len / 2)

  params.full_accuracy_test = {'dataset_folder': params.datasets2use['test'][0],
                               'labels': dataset_prepare.model_net_labels,
                               'min_max_faces2use': params.test_min_max_faces2use,
                               'n_walks_per_model': 16 * 4,
                               }

  # Parameters to recheck:
  params.iters_to_train = 500e3
  params.net_input = ['xyz', 'v_normals']
  params.walk_alg = 'random_global_jumps'  # no_jumps / global_jumps

  # Attention params for new network:
  if mode == 'Attention':
    params.optimizer_type = 'cycle'  # 'transformer'
    params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-7,
                                      'maximal_learning_rate': 5e-5,
                                      'step_size': 10000})
    params.net = 'HierTransformer'
    # params.walk_alg = 'constant_global_jumps_new'  # 'constant_global_jumps_bfs'   #'constant_global_jumps'
    params.walk_alg = 'constant_global_jumps'  #'random_global_jumps'  # 'constant_global_jumps'
    params.net_params = EasyDict({'num_layers': 3,  # number of stacked layers in transformer
                                  'num_scales': 1,
                                  'd_model': 256,  # dimension of feature vector
                                  'num_heads': 8,  # number of parallel multi-heads
                                  'dff': 1024,  # dimension of feed-forward at the end of each encoder\decoder unit
                                  'input_vocab_size': 20,  # check if relevant should be deleted later
                                  'out_features': 256,
                                  'jump_every_k': 20,
                                  'pe_input': 20,  # maximum positional encoding for input - ?! check effect
                                  'pe_target': 20,  # maximum positional encoding for target - ?! check effect
                                  'global_dim_mult': 1,
                                  'rate': 0.0,
                                  'pooling': False,
                                  'concat_xyz': False,
                                  'recurrent': False
                                  })
    params.logdir += '_attention'
  # params.train_max_size_per_class = 20
  elif mode == 'rnnAttention':
    params.net = 'rnnAttention'
    params.logdir += '_rnnAttention'
  params.logdir += '_{}'.format(params.seq_len)
  params.logdir += '_{}'.format(mode)
  return params