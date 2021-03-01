import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import sys
from easydict import EasyDict

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import rnn_model
import attention_model
import dataset
import dataset_prepare
import utils
import params_setting
from tqdm import tqdm


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class my_scheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, init_lr, decay_epochs, train_epoch_size, decay_rate=0.01, min_lr=8e-7):
    super(my_scheduler, self).__init__()
    self.decay_epochs = decay_epochs
    self.train_size = train_epoch_size
    self.decay_rate = decay_rate
    self.cur_lr = tf.cast(init_lr, tf.float32)
    self.min_lr = min_lr

  @tf.function
  def __call__(self, step):
    epoch = step // self.train_size
    if epoch < self.decay_epochs:
      return self.cur_lr
    else:
      return tf.maximum(self.cur_lr * tf.math.exp(self.decay_rate * (self.decay_epochs - epoch)), self.min_lr)


def train_val(params):
  '''
  TODO: Adding corriculum learning for seq_len (starting at 200, 400, 800)
  :param params:
  :return:
  '''
  if params.train_multiwalk_head:
    utils.next_iter_to_keep = params.cycle_opt_prms.step_size  # 2000
  else:
    utils.next_iter_to_keep = params.cycle_opt_prms.step_size  # 10000
  print(utils.color.BOLD + utils.color.RED + 'params.logdir :::: ', params.logdir, utils.color.END)
  print(utils.color.BOLD + utils.color.RED, os.getpid(), utils.color.END)
  utils.backup_python_files_and_params(params)
  if not os.path.exists(os.path.join(params.logdir, "results")):
    os.makedirs(os.path.join(params.logdir, "results"))
  # Set up datasets for training and for test
  # -----------------------------------------


  def get_train_set(params):
    train_datasets = []
    train_ds_iters = []
    for i in range(len(params.datasets2use['train'])):
      this_train_dataset, n_trn_items = dataset.tf_mesh_dataset(params, params.datasets2use['train'][i],
                                                                mode=params.network_tasks[i], size_limit=params.train_dataset_size_limit,
                                                                shuffle_size=100, min_max_faces2use=params.train_min_max_faces2use,
                                                                max_size_per_class=params.train_max_size_per_class, min_dataset_size=64*32,
                                                                data_augmentation=params.train_data_augmentation)
      print('Train Dataset size:', n_trn_items)
      train_ds_iters.append(iter(this_train_dataset.repeat()))
      train_datasets.append(this_train_dataset)
    train_epoch_size = max(8, int(n_trn_items / params.n_walks_per_model / params.batch_size))
    print('train_epoch_size:', train_epoch_size)
    return train_datasets, train_epoch_size, train_ds_iters

  def get_test_set(params):
    test_dataset, n_tst_items = dataset.tf_mesh_dataset(params, params.datasets2use['test'][0],
                                                        mode=params.network_tasks[0],
                                                        size_limit=params.test_dataset_size_limit,
                                                        shuffle_size=100,
                                                        min_max_faces2use=params.test_min_max_faces2use,
                                                        )
    print(' Test Dataset size:', n_tst_items)
    return test_dataset, n_tst_items


  train_datasets, train_epoch_size, train_ds_iters = get_train_set(params)


  if params.datasets2use['test'] is None:
    test_dataset = None
    n_tst_items = 0
  else:
    test_dataset, n_tst_items = get_test_set(params)

  # Set up RNN model and optimizer
  # ------------------------------
  if params.net_start_from_prev_net is not None:
    init_net_using = params.net_start_from_prev_net
  else:
    init_net_using = None

  if params.optimizer_type == 'adam':
    lr_schedule = my_scheduler(init_lr=params.learning_rate[0], decay_epochs=20, train_epoch_size=train_epoch_size*20)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=params.gradient_clip_th)
  elif params.optimizer_type == 'cycle':
    @tf.function
    def _scale_fn(x):
      x_th = 400e3 / params.cycle_opt_prms.step_size
      if x < x_th:
        return 1.0
      else:
        return 0.5
    lr_schedule = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params.cycle_opt_prms.initial_learning_rate,
                                                      maximal_learning_rate=params.cycle_opt_prms.maximal_learning_rate,
                                                      step_size=params.cycle_opt_prms.step_size,
                                                      scale_fn=_scale_fn, scale_mode="cycle", name="MyCyclicScheduler")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=params.gradient_clip_th)
  elif params.optimizer_type == 'sgd':
    optimizer = tf.keras.optimdizers.SGD(lr=params.learning_rate[0], decay=0, momentum=0.9, nesterov=True, clipnorm=params.gradient_clip_th)
  else:
    raise Exception('optimizer_type not supported: ' + params.optimizer_type)

  if params.net == 'RnnWalkNet':
    dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim, init_net_using, optimizer=optimizer)
  elif params.net == 'RnnAttention':
    dnn_model = rnn_model.AttentionWalkNet(params, params.n_classes, params.net_input_dim, init_net_using, optimizer=optimizer)
  elif params.net == 'HierTransformer':
    dnn_model = attention_model.WalkHierTransformer(**params.net_params, params=params, optimizer=optimizer,
                                                    model_fn=init_net_using)
  # Other initializations
  # ---------------------
  time_msrs = {}
  time_msrs_names = ['train_step', 'get_train_data', 'test']
  for name in time_msrs_names:
    time_msrs[name] = 0
  seg_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='seg_train_accuracy')

  train_log_names = ['seg_loss', 'triplet_center_loss']
  train_logs = {name: tf.keras.metrics.Mean(name=name) for name in train_log_names}
  train_logs['seg_train_accuracy'] = seg_train_accuracy

  # Train / test functions
  # ----------------------
  if params.last_layer_actication is None:
    seg_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  else:
    seg_loss = tf.keras.losses.SparseCategoricalCrossentropy()


  if params.train_multiwalk_head:
    train_vars = dnn_model.trainable_variables[-6:]
    # train_vars = dnn_model.trainable_variables
  else:
    train_vars = dnn_model.trainable_variables

  if params.triplet_loss:
    triplet_center_loss = rnn_model.TCL(params.n_classes, params.triplet_dim)
    if params.centers_weights:
      dummy_input = tf.ones((4, params.triplet_dim))
      _ = triplet_center_loss(dummy_input, tf.constant([1, 1, 2, 2], dtype=tf.int32))
      triplet_center_loss.load_weights(params.centers_weights)
    train_vars += triplet_center_loss.trainable_variables
    len_triplet_vars = len(triplet_center_loss.trainable_variables)
    from evaluation import calculate_map_auc
    train_logs['mAP'] = tf.keras.metrics.Mean(name='mAP')
    train_logs['AUC'] = tf.keras.metrics.Mean(name='AUC')
    train_log_names += ['mAP', 'AUC']

  @tf.function
  def train_step(model_ftrs_, labels_, one_label_per_model):
    sp = model_ftrs_.shape
    model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
    with tf.GradientTape() as tape:
      if one_label_per_model:
        if params.cross_walk_attn:
          labels = tf.reshape(tf.transpose(tf.stack((labels_,) * 1)), (-1,))
        else:
          labels = tf.reshape(tf.transpose(tf.stack((labels_,)*params.n_walks_per_model)),(-1,))
        predictions, ftrs = dnn_model(model_ftrs, classify='both')
        if params.triplet_loss:
          triplet_loss = triplet_center_loss(ftrs, labels, margin=5.0)
      else:
        labels = tf.reshape(labels_, (-1, sp[-2]))
        skip = params.min_seq_len if 'RnnWalkNet' in params.net else 0
        predictions = dnn_model(model_ftrs)[:, skip:]
        labels = labels[:, skip:]
      if hasattr(params, 'ignore_label'):
        labels = tf.reshape(labels, [-1])
        predictions = tf.reshape(predictions, [-1, params.n_classes])
        indices = tf.squeeze(tf.where(tf.not_equal(labels, params.ignore_label)), 1)
        labels = tf.cast(tf.gather(labels, indices), tf.int32)
        predictions = tf.gather(predictions, indices)

      seg_train_accuracy(labels, predictions)
      if hasattr(params, 'class_weights'):
        beta = 0.9
        effective_num = 1.0 - np.power(beta, [x / min(params.class_weights) for x in np.asarray(params.class_weights)])
        weights = (1.0-beta)/ np.array(effective_num)
        weights = weights / np.sum(weights) * int(params.n_classes)
      else:
        weights = tf.ones((params.n_classes))
      weights = tf.gather(weights, labels)
      loss = seg_loss(labels, predictions, weights)
      loss += tf.reduce_sum(dnn_model.losses)
      if params.triplet_loss and not tf.math.is_nan(triplet_loss):
        loss += 0.1 * triplet_loss
        train_logs['triplet_center_loss'](triplet_loss)
    gradients = tape.gradient(loss, train_vars)
    # if params.triplet_loss:
    #   gradients[-len_triplet_vars] = [100 * x for x in gradients[-len_triplet_vars]] # according to TCL, centers learning rate is ~ x 100
    # if params.train_multiwalk_head:
    #   gradients[-6:] = gradients[-6:] * 10
    optimizer.apply_gradients(zip(gradients, train_vars))

    train_logs['seg_loss'](loss)

    return loss

  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  @tf.function
  def test_step(model_ftrs_, labels_, one_label_per_model):
    sp = model_ftrs_.shape
    model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
    if one_label_per_model:
      if params.cross_walk_attn:
        labels = tf.reshape(tf.transpose(tf.stack((labels_,) * 1)), (-1,))
      else:
        labels = tf.reshape(tf.transpose(tf.stack((labels_,) * params.n_walks_per_model)), (-1,))
      predictions = dnn_model(model_ftrs, training=False)
    else:
      labels = tf.reshape(labels_, (-1, sp[-2]))
      skip = params.min_seq_len
      predictions = dnn_model(model_ftrs, training=False)[:, skip:]
      labels = labels[:, skip:]
    if hasattr(params, 'ignore_label'):
      labels = tf.reshape(labels, [-1])
      predictions = tf.reshape(predictions, [-1, params.n_classes])
      indices = tf.squeeze(tf.where(tf.not_equal(labels, params.ignore_label)), 1)
      labels = tf.cast(tf.gather(labels, indices), tf.int32)
      predictions = tf.gather(predictions, indices)

    best_pred = tf.math.argmax(predictions, axis=-1)
    test_accuracy(labels, predictions)
    confusion = tf.math.confusion_matrix(labels=tf.reshape(labels, (-1,)), predictions=tf.reshape(best_pred, (-1,)),
                                         num_classes=params.n_classes)
    return confusion
  # -------------------------------------

  # Loop over training EPOCHs
  # -------------------------
  one_label_per_model = params.network_task == 'classification'
  next_iter_to_log = 0
  e_time = 0
  accrcy_smoothed = tb_epoch = last_loss = None
  all_confusion = {}
  half_flag = 0
  last_quarter_flag = 0
  with tf.summary.create_file_writer(params.logdir).as_default():
    epoch = 0
    while optimizer.iterations.numpy() < params.iters_to_train + train_epoch_size * 2:
      # '''
      # Corriculum - we increase seq_len gradually as we advance in training, during 2 steps: 50%, 75%
      # '''
      # if optimizer.iterations.numpy() > int(3 * params.iters_to_train / 4) and not last_quarter_flag:
      #   last_quarter_flag = 1
      #   print('Updating seq_len to 800')
      #   params.seq_len = 800
      #   train_datasets, train_epoch_size, train_ds_iters = get_train_set(params)
      #   test_dataset, n_tst_items = get_test_set(params)
      # elif optimizer.iterations.numpy() > params.iters_to_train / 2 and not half_flag:
      #   half_flag = 1
      #   print('Updating seq_len to 400')
      #   params.seq_len = 400
      #   train_datasets, train_epoch_size = get_train_set(params)
      #   test_dataset, n_tst_items, train_ds_iters = get_test_set(params)


      epoch += 1
      str_to_print = str(os.getpid()) + ') Epoch' + str(epoch) + ', iter ' + str(optimizer.iterations.numpy())
      # if epoch == 1 and params.train_multiwalk_head:
      #   # Do single full test to see our starting point is OK
      #   old_iter2keep = utils.next_iter_to_keep
      #   dnn_model.cross_walk_attn = False
      #   utils.save_model_if_needed(optimizer.iterations, dnn_model, params, override=True)
      #   utils.next_iter_to_keep = old_iter2keep
      #   dnn_model.cross_walk_attn = True
      # Save some logs & infos
      if optimizer.iterations.numpy() >= utils.next_iter_to_keep and params.triplet_loss:
        triplet_center_loss.save_weights(params.logdir, optimizer.iterations.numpy())
      utils.save_model_if_needed(optimizer.iterations, dnn_model, params)
      if tb_epoch is not None:
        e_time = time.time() - tb_epoch
        tf.summary.scalar('time/one_epoch', e_time, step=optimizer.iterations)
        tf.summary.scalar('time/av_one_trn_itr', e_time / n_iters, step=optimizer.iterations)
        for name in time_msrs_names:
          if time_msrs[name]:  # if there is something to save
            tf.summary.scalar('time/' + name, time_msrs[name], step=optimizer.iterations)
            time_msrs[name] = 0
      tb_epoch = time.time()
      n_iters = 0
      tf.summary.scalar(name="train/learning_rate", data=optimizer._decayed_lr(tf.float32), step=optimizer.iterations)
      tf.summary.scalar(name="mem/free", data=utils.check_mem_and_exit_if_full(), step=optimizer.iterations)

      # Train one EPOC
      str_to_print += '; LR: ' + str(optimizer._decayed_lr(tf.float32).numpy())
      train_logs['seg_loss'].reset_states()
      tb = time.time()
      for iter_db in tqdm(range(train_epoch_size)):
        for dataset_id in range(len(train_datasets)):
          name, model_ftrs, labels = train_ds_iters[dataset_id].next()
          dataset_type = utils.get_dataset_type_from_name(name)
          if params.learning_rate_dynamics != 'stable':
            utils.update_lerning_rate_in_optimizer(0, params.learning_rate_dynamics, optimizer, params)
          time_msrs['get_train_data'] += time.time() - tb
          n_iters += 1
          tb = time.time()
          if params.train_loss[dataset_id] == 'cros_entr':
            train_step(model_ftrs, labels, one_label_per_model=one_label_per_model)
            loss2show = 'seg_loss'
          else:
            raise Exception('Unsupported loss_type: ' + params.train_loss[dataset_id])
          time_msrs['train_step'] += time.time() - tb
          tb = time.time()
        if iter_db == train_epoch_size - 1:
          str_to_print += ', TrnLoss: ' + str(round(train_logs[loss2show].result().numpy(), 2))

      # Dump training info to tensorboard
      if optimizer.iterations >= next_iter_to_log:
        for k, v in train_logs.items():
          if v.count.numpy() > 0:
            tf.summary.scalar('train/' + k, v.result(), step=optimizer.iterations)
            v.reset_states()
        next_iter_to_log += params.log_freq

      # Run test on part of the test set
      if test_dataset is not None:
        n_test_iters = 0
        tb = time.time()
        for name, model_ftrs, labels in test_dataset:
          n_test_iters += model_ftrs.shape[0]
          if n_test_iters > params.n_models_per_test_epoch:
            break
          confusion = test_step(model_ftrs, labels, one_label_per_model=one_label_per_model)
          dataset_type = utils.get_dataset_type_from_name(name)
          if dataset_type in all_confusion.keys():
            all_confusion[dataset_type] += confusion
          else:
            all_confusion[dataset_type] = confusion
        # Dump test info to tensorboard
        if accrcy_smoothed is None:
          accrcy_smoothed = test_accuracy.result()
        accrcy_smoothed = accrcy_smoothed * .9 + test_accuracy.result() * 0.1
        tf.summary.scalar('test/accuracy_' + dataset_type, test_accuracy.result(), step=optimizer.iterations)
        tf.summary.scalar('test/accuracy_smoothed', accrcy_smoothed, step=optimizer.iterations)
        str_to_print += ', test/accuracy_' + dataset_type + ': ' + str(round(test_accuracy.result().numpy(), 2))
        test_accuracy.reset_states()
        time_msrs['test'] += time.time() - tb

      str_to_print += ', time: ' + str(round(e_time, 1))
      print(str_to_print) 

  return last_loss


def run_one_job(job, job_part):
  # Classifications
  job = job.lower()
  if job == 'modelnet40_retrieval' or job == 'modelnet_retrieval':
    params = params_setting.modelnet_params(job_part, split=1)

  if job == 'modelnet40' or job == 'modelnet':
    params = params_setting.modelnet_params(job_part, short=False)

  if job == 'shrec11':
    params = params_setting.shrec11_params(job_part, mode='Multiwalk')   #'Multiwalk'
    # params.net_start_from_prev_net='/home/ran/mesh_walker/runs_compare/0207-10.12.2020..11.34__future_raw_800/ckpt-19.index'

  if job == 'cubes':
    params = params_setting.cubes_params(job_part)

  # Semantic Segmentations
  if job == 'human_seg':
    params = params_setting.human_seg_params(job_part)

  if job == 'coseg':
    params = params_setting.coseg_params(job_part)   #  job_part can be : 'aliens' or 'chairs' or 'vases'

  if job == 'future':
    params = params_setting.future_params(job_part)
  train_val(params)


def get_all_jobs():
  jobs      = [
    'shrec11', 'shrec11', 'shrec11',
    'shrec11', 'shrec11', 'shrec11',
    'coseg', 'coseg', 'coseg',
    'human_seg',
    'cubes',
    'modelnet40',
  ]
  job_parts = [
    '10-10_A', '10-10_B', '10-10_C',
    '16-04_A', '16-04_B', '16-04_C',
    'aliens', 'vases', 'chairs',
    None,
    None,
    None,
  ]

  return jobs, job_parts

if __name__ == '__main__':
  np.random.seed(0)
  utils.config_gpu()

  # job = 'modelnet'
  # job_part = 'rnn' #'Attention'   # 'Multiwalk'  #'
  #
  # job = 'future'
  # job_part = 'rnn' #'Attention'   # 'Multiwalk'  #'
  # job = 'cubes'
  # job_part = 'Multiwalk' #'Attention'   # 'Multiwalk'

  # job = 'human_seg'
  # job_part = 'Multiwalk'  # 'Attention'   # 'Multiwalk'

  # job = 'modelnet40'
  job = 'modelnet40'
  job_part = 'Multiwalk'  # 'Attention'   # 'Multiwalk'  #'rnn'

  # job = 'modelnet40_retrieval'
  # job_part = 'rnn'  # 'Attention'   # 'Multiwalk'  #'rnn'

  # job = 'shrec11'
  # job_part = '16-04_C'  # 'Attention'   # 'Multiwalk'  #'rnn'

  if len(sys.argv) > 1:
    job = sys.argv[1].lower()
  if len(sys.argv) > 2:
    job_part = sys.argv[2].lower()

  if job.lower() == 'all':
    jobs, job_parts = get_all_jobs()
    for job_, job_part in zip(jobs, job_parts):
      run_one_job(job_, job_part)
  else:
    run_one_job(job, job_part)


  #
  # # params = params_setting.modelnet_params('Attention')   #  'rnnWalk'   'Attention' 'rnnAttention'
  #
  # # params = params_setting.scannet_params('rnnWalk')
  # params = params_setting.scannet_params('Attention')
  # # import glob
  # # params.net_start_from_prev_net = glob.glob('/home/ran/mesh_walker/runs_aug_360_must/0114-10.11.2020..15.51__scannet_v2/*.keras')[-1]
  # train_val(params)


