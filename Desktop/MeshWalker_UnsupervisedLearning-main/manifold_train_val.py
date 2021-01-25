import os
import time
import sys
from easydict import EasyDict

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import rnn_model
import dataset
import utils
import params_setting
# Amir
import split_data_classes_folders
recon_train = True
import argparse
import random

def print_enters(to_print):
  print("\n\n\n\n")
  print(to_print)
  print("\n\n\n\n")

def label_to_one_hot(labels: tf.Tensor, params, alpha=0, shift_size=1):
  if alpha == 0:
    return tf.one_hot(indices=labels, depth=params.n_classes)
  else:
    return get_manifold_labels(tf.one_hot(indices=labels, depth=params.n_classes), alpha, shift_size)

def get_manifold_labels(one_hot_labels, alpha, shift_size):
  orig_labels = one_hot_labels
  shifted_labels = tf.roll(orig_labels, shift=[shift_size, 0], axis=[0, 1])
  manifold_labels = orig_labels * alpha + shifted_labels * (1 - alpha)
  return manifold_labels


def train_val(params):
  utils.next_iter_to_keep = 10000
  print(utils.color.BOLD + utils.color.RED + 'params.logdir :::: ', params.logdir, utils.color.END)
  print(utils.color.BOLD + utils.color.RED, os.getpid(), utils.color.END)
  utils.backup_python_files_and_params(params)

  # Set up datasets_processed for training and for test
  # -----------------------------------------
  train_datasets = []
  train_ds_iters = []
  max_train_size = 0
  for i in range(len(params.datasets2use['train'])):
    # Gals changes
    this_train_dataset, n_trn_items = dataset.tf_mesh_dataset(params, params.datasets2use['train'][i],
                                                              mode=params.network_tasks[i],
                                                              size_limit=params.train_dataset_size_limit,
                                                              shuffle_size=100,
                                                              min_max_faces2use=params.train_min_max_faces2use,
                                                              max_size_per_class=params.train_max_size_per_class,
                                                              min_dataset_size=128,
                                                              data_augmentation=params.train_data_augmentation)
    #Amir
    if config['generate_csv_file2label'] == True:
        split_data_classes_folders.tf_mesh_dataset(params, params.datasets2use['train'][i],
                                                   min_max_faces2use=params.train_min_max_faces2use,
                                                   data_augmentation=params.train_data_augmentation, name="train")

    # End
    print('Train Dataset size:', n_trn_items)
    train_ds_iters.append(iter(this_train_dataset.repeat()))
    train_datasets.append(this_train_dataset)

    max_train_size = max(max_train_size, n_trn_items)
  train_epoch_size = max(8, int(max_train_size / params.n_walks_per_model / params.batch_size))
  print('train_epoch_size:', train_epoch_size)
  if params.datasets2use['test'] is None:
    test_dataset = None
    n_tst_items = 0
  else:
    # Gals changes
    test_dataset, n_tst_items = dataset.tf_mesh_dataset(params, params.datasets2use['test'][0],
                                                        mode=params.network_tasks[0],
                                                        size_limit=params.test_dataset_size_limit,
                                                        shuffle_size=100,
                                                        min_max_faces2use=params.test_min_max_faces2use)
    test_ds_iter = iter(test_dataset.repeat())
    # Amir
    if config['generate_csv_file2label'] == True:
      split_data_classes_folders.tf_mesh_dataset(params, params.datasets2use['train'][i],
                                                 min_max_faces2use=params.train_min_max_faces2use,
                                                 data_augmentation=params.train_data_augmentation, name = "test")

    # End
  print(' Test Dataset size:', n_tst_items)

  # Set up RNN model and optimizer
  # ------------------------------
  if params.net_start_from_prev_net is not None:
    init_net_using = params.net_start_from_prev_net
  else:
    init_net_using = None

  if params.optimizer_type == 'adam':
    optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate[0], clipnorm=params.gradient_clip_th)
  elif params.optimizer_type == 'cycle':
    @tf.function
    def _scale_fn(x):
      x_th = 500e3 / params.cycle_opt_prms.step_size
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
    optimizer = tf.keras.optimizers.SGD(lr=params.learning_rate[0], decay=0, momentum=0.9, nesterov=True,
                                        clipnorm=params.gradient_clip_th)
  else:
    raise Exception('optimizer_type not supported: ' + params.optimizer_type)

  if config['use_pretrained_model'] is True:
    model_fn = tf.train.latest_checkpoint(config['trained_model'])
    if params.net == 'RnnWalkNet':
      dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, 3, model_fn,
                                     model_must_be_load=True, dump_model_visualization=False)
    elif params.net == "Manifold_RnnWalkNet":
      dnn_model = rnn_model.RnnManifoldWalkNet(params, params.n_classes, 3, model_fn,
                                     model_must_be_load=True, dump_model_visualization=False)
    elif params.net == 'Unsupervised_RnnWalkNet':
      dnn_model = rnn_model.Unsupervised_RnnWalkNet(params, params.n_classes, 3, model_fn,
                                     model_must_be_load=True, dump_model_visualization=False)
  else:
    if params.net == 'RnnWalkNet':
      dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim, init_net_using,
                                       optimizer=optimizer)
    elif params.net == "Manifold_RnnWalkNet":
      dnn_model = rnn_model.RnnManifoldWalkNet(params, params.n_classes, params.net_input_dim, init_net_using,
                                       optimizer=optimizer)
    elif params.net == 'Unsupervised_RnnWalkNet':
      dnn_model = rnn_model.Unsupervised_RnnWalkNet(params, params.n_classes, params.net_input_dim, init_net_using,
                                       optimizer=optimizer)

  # Other initializations
  # ---------------------
  time_msrs = {}
  time_msrs_names = ['train_step', 'get_train_data', 'test']
  for name in time_msrs_names:
    time_msrs[name] = 0
  seg_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='seg_train_accuracy')

  train_log_names = ['seg_loss']
  train_logs = {name: tf.keras.metrics.Mean(name=name) for name in train_log_names}
  train_logs['seg_train_accuracy'] = seg_train_accuracy

  # Train / test functions
  # ----------------------
  if params.last_layer_actication is None:
    if params.net == 'Manifold_RnnWalkNet':
      seg_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    else:
      seg_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if params.train_loss == ['triplet']:
      seg_loss = tfa.losses.TripletSemiHardLoss(from_logits=True)
  else:
    if params.net == 'Manifold_RnnWalkNet':
      seg_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    else:
      seg_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    if params.train_loss == ['triplet']:
      seg_loss = tfa.losses.TripletSemiHardLoss()

  #@tf.function
  def train_step(model_ftrs_, labels_, one_label_per_model):
    sp = model_ftrs_.shape
    model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
    with tf.GradientTape() as tape:
      if one_label_per_model:
        # Gal, when multiple walks are per walk, the labels are like this (for example if 2 walks per model):
        # labels = [1,1,7,7,8,8,3,3,16,16...]
        labels = tf.reshape(tf.transpose(tf.stack((labels_,)*params.n_walks_per_model)),(-1,))
        if params.net == 'Manifold_RnnWalkNet':
          predictions = dnn_model(model_ftrs, alpha, config['shift_size'])
        else:
          predictions = dnn_model(model_ftrs)
      else:
        labels = tf.reshape(labels_, (-1, sp[-2]))
        skip = params.min_seq_len
        predictions = dnn_model(model_ftrs)[:, skip:]
        labels = labels[:, skip + 1:]

      if params.net == 'Manifold_RnnWalkNet':
        labels = label_to_one_hot(labels, params)
      else:
        seg_train_accuracy(labels, predictions)
      loss = seg_loss(labels, predictions)
      loss += tf.reduce_sum(dnn_model.losses)

    gradients = tape.gradient(loss, dnn_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, dnn_model.trainable_variables))

    train_logs['seg_loss'](loss)

    return loss

  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  #@tf.function
  def test_step(model_ftrs_, labels_, one_label_per_model):
    sp = model_ftrs_.shape
    model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
    if one_label_per_model:
      labels = tf.reshape(tf.transpose(tf.stack((labels_,) * params.n_walks_per_model)), (-1,))
      if params.net == 'Manifold_RnnWalkNet':
        predictions = dnn_model(model_ftrs, alpha=0, shift_size=config['shift_size'])
      else:
        predictions = dnn_model(model_ftrs)
    else:
      labels = tf.reshape(labels_, (-1, sp[-2]))
      skip = params.min_seq_len
      predictions = dnn_model(model_ftrs)[:, skip:]
      labels = labels[:, skip + 1:]

    if params.net == 'Manifold_RnnWalkNet':
      labels = label_to_one_hot(labels, params)
    else:
      test_accuracy(labels, predictions)
    best_pred = tf.math.argmax(predictions, axis=-1)

    confusion = None
    #Amir
    # the confusion had to recive zero label, that's a problem
    # So for now I'm skipping this part if we are just doing reconstruction
    if recon_train == False:
      confusion = tf.math.confusion_matrix(labels=tf.reshape(labels, (-1,)), predictions=tf.reshape(best_pred, (-1,)),
                                           num_classes=params.n_classes)

    return confusion
  # -------------------------------------

  # Loop over training EPOCHs
  # -------------------------
  one_label_per_model = params.one_label_per_model
  next_iter_to_log = 0
  e_time = 0
  accrcy_smoothed = tb_epoch = last_loss = None
  all_confusion = {}
  with tf.summary.create_file_writer(params.logdir).as_default():
    epoch = 0
    while optimizer.iterations.numpy() < params.iters_to_train + train_epoch_size * 2:
      epoch += 1
      str_to_print = str(os.getpid()) + ') Epoch' + str(epoch) + ', iter ' + str(optimizer.iterations.numpy())

      # Save some logs & infos
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
      gpu_tmpr = utils.get_gpu_temprature()
      if gpu_tmpr > 95:
        print('GPU temprature is too high!!!!!')
        exit(0)
      tf.summary.scalar(name="mem/gpu_tmpr", data=gpu_tmpr, step=optimizer.iterations)

      # Train one EPOC
      alpha = 1 # manifold variable
      if optimizer.iterations.numpy() % config['non_zero_ratio'] == 0:
        alpha = 1 #random.uniform(0, 0.5)

      str_to_print += '; LR: ' + str(optimizer._decayed_lr(tf.float32))
      train_logs['seg_loss'].reset_states()
      tb = time.time()
      for iter_db in range(train_epoch_size):
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
          elif params.train_loss[dataset_id] == 'triplet':
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
        #for name, model_ftrs, labels in test_dataset:
        for i in range(n_tst_items):
          name, model_ftrs, labels = test_ds_iter.next()

          n_test_iters += model_ftrs.shape[0]
          if n_test_iters > params.n_models_per_test_epoch:
            break
          confusion = test_step(model_ftrs, labels, one_label_per_model=one_label_per_model)
          # Amir - added the case that confusion is none as a result of recon training
          if confusion is not None:
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

def run_one_job(job, job_part, network_task):
  # Classifications
  job = job.lower()
  if job == 'modelnet40' or job == 'modelnet':
    params = params_setting.modelnet_params(network_task)

  if job == 'shrec11':
    params = params_setting.shrec11_params(job_part, network_task)

  if job == 'cubes':
    params = params_setting.cubes_params(network_task)

  # Semantic Segmentations
  if job == 'human_seg':
    params = params_setting.human_seg_params(network_task)

  if job == 'coseg':
    params = params_setting.coseg_params(job_part, network_task)   #  job_part can be : 'aliens' or 'chairs' or 'vases'


  # train only on a subset of the classes
  if job == 'shrec11':
    params.classes_indices_to_use = [*range(30)]
    # Amir - If I ever want to run only on 2 classes - this is something I should do
    #params.classes_indices_to_use = (params.classes_indices_to_use)[0:2]
    if config['trained_only_2_classes'] == True:
      #params.classes_indices_to_use = (params.classes_indices_to_use)[0:2]
      first_label = min(config['source_label'], config['target_label'])
      sec_label = max(config['source_label'], config['target_label'])
      params.classes_indices_to_use = [first_label, sec_label] #list([15, 25]) #(params.classes_indices_to_use[15],params.classes_indices_to_use[25]) # [15, 25]  # 15 - horse, 25 - camel
    #params.n_classes = len(params.classes_indices_to_use)

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
    '16-04_a', '16-4_B', '16-4_C',
    'aliens', 'vases', 'chairs',
    None,
    None,
    None,
  ]

  return jobs, job_parts

if __name__ == '__main__':
  np.random.seed(0)
  utils.config_gpu()

  # get hyper params from yaml
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default='recon_config.yaml', help='Path to the config file.')
  opts = parser.parse_args()
  config = utils.get_config(opts.config)

  job = config['job']
  #job_part = '10-10_A'
  job_part = config['job_part'] #16-04_a'

  # choose network task from: 'features_extraction', 'unsupervised_classification', 'semantic_segmentation', 'classification'. 'manifold_classification'
  network_task = 'manifold_classification'

  """if len(sys.argv) > 1:
    job = sys.argv[1].lower()
  else:
    print("Please specify the job ! exiting..")
    exit(0)
  if len(sys.argv) > 2:
    job_part = sys.argv[2].lower()"""

  if job.lower() == 'all':
    jobs, job_parts = get_all_jobs()
    for job_, job_part in zip(jobs, job_parts):
      run_one_job(job_, job_part, network_task)
  else:
    run_one_job(job, job_part, network_task)

