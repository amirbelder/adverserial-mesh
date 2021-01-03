"""Training loop for MeshWalker on dancer segmentation."""
# pylint: disable=missing-function-docstring

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tqdm  # pylint: disable=g-bad-import-order

import augment  # pylint: disable=g-bad-import-order
import helpers  # pylint: disable=g-bad-import-order
import rnn_model_tfg
import dataset_tfg
import utils_tfg

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

utils_tfg.config_gpu()

parser = helpers.ArgumentParser()
parser.add("--walk_length", 800, help="length of walk")
parser.add("--augment", True, help="use augmentations")
parser.add("--one_label_per_model", False, help="True for classification task, False for segmentation")

parser.add("--pretrained_model_path", '', help="path to pretrained model - if cleared, atart from scratch")

parser.add("--only_evaluate", False, help="to run evaluation only")
parser.add("--tb_every", 100, help="tensorboard frequency (iterations)")
parser.add("--ev_every", 100, help="evaluation frequency (iterations)")
parser.add("--ev_full_every", 1000, help="full evaluation frequency (iterations)")
parser.add("--ev_number_of_meshes", 10, help="number of meshes to evaluate. If set to 0, all test dataset will be used (will slow the run).")

parser.add("--tqdm", True, help="enable the progress bar")

FLAGS = parser.parse_args()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

dataset_name = 'dancer'
ds_train, classification_task, n_classes = dataset_tfg.get_dataset(dataset_name, is_training=True)
ds_test, _, _ = dataset_tfg.get_dataset(dataset_name, is_training=False)
FLAGS.one_label_per_model = classification_task     # For dancer, each mesh has many labels - as number of its vertices - so "one_label_per_model" will be false

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

lr_schedule = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=1e-6,
                                                  maximal_learning_rate=2e-5,
                                                  step_size=10000,
                                                  scale_fn=(lambda _: 1.0), scale_mode="cycle", name="MyCyclicScheduler")
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

skip = int(FLAGS.walk_length / 2)
model = rnn_model_tfg.RnnWalkNet(FLAGS, classes=n_classes, net_input_dim=3, model_fn=None, optimizer=optimizer)
if FLAGS.pretrained_model_path != '':
  model.load_weights(FLAGS.pretrained_model_path)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

seg_loss = tf.keras.losses.SparseCategoricalCrossentropy()
@tf.function
def wrapped_tf_function(walks, labels):
  """Performs one step of minimization of the loss."""

  # --- augmentation
  if FLAGS.augment:
    walks = augment.rotate(walks)
    #walks = augment.jitter(walks)

  # --- training
  with tf.GradientTape() as tape:
    logits = model(walks, training=True)
    if FLAGS.one_label_per_model:
      loss = seg_loss(tf.repeat(labels, logits.shape[0]), logits)
    else:
      loss = seg_loss(labels[:, skip + 1:], logits[:, skip:])
  variables = model.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return loss


def train(example):
  """Performs one step of minimization of the loss and populates the summary."""
  walks, walks_vertices = dataset_tfg.example_to_walks_representation(example, FLAGS.walk_length)
  if 'labels' in example:
    labels = example['labels'][0].numpy()[walks_vertices]
  else:
    labels = example['mesh_label'][0].numpy()
  step = optimizer.iterations.numpy()

  # --- optimize
  loss = wrapped_tf_function(walks, labels)
  if step % FLAGS.tb_every == 0:
    tf.summary.scalar(name="loss_" + dataset_name, data=loss, step=step)
    tf.summary.scalar(name="learning_rate", data=optimizer._decayed_lr(tf.float32), step=step)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def evaluate(one_mesh=False):
  """Identify the best accuracy reached during training."""
  step = optimizer.iterations.numpy()
  if "best_accuracy" not in evaluate.__dict__:
    evaluate.best_accuracy = 0
  if step % FLAGS.ev_every != 0 and step % FLAGS.ev_full_every != 0:
    return evaluate.best_accuracy
  model.save_weights(FLAGS.logdir, step=step, keep=False)
  acc_per_walk = tf.keras.metrics.SparseCategoricalAccuracy()
  full_accuracy = []
  n_walks_per_mesh = 4
  n_iters_per_mesh = 64 if (step % FLAGS.ev_full_every == 0) else 1
  total = FLAGS.ev_number_of_meshes
  if total <= 0:
    total = np.inf
  print('\nEvaluating!\n')
  for example in ds_test:
    all_walks_vertices = None
    all_logits = None
    for _ in range(n_iters_per_mesh):
      walks, walks_vertices = dataset_tfg.example_to_walks_representation(example, FLAGS.walk_length, n_walks_per_mesh=n_walks_per_mesh)
      logits = model(walks, training=False)
      if FLAGS.one_label_per_model:
        labels = np.repeat(example['mesh_label'][0].numpy(), logits.shape[0])
        acc_per_walk.update_state(labels, logits)
      else:
        labels = example['labels'][0].numpy()[walks_vertices]
        acc_per_walk.update_state(labels[:, skip + 1:], logits[:, skip:])
      if all_walks_vertices is None:
        all_walks_vertices = walks_vertices
        all_logits = logits
      else:
        all_walks_vertices = np.vstack((all_walks_vertices, walks_vertices))
        all_logits = np.vstack((all_logits, logits))
    if not FLAGS.one_label_per_model and (step % FLAGS.ev_full_every == 0):
      this_full_accuracy = utils_tfg.posttprocess_and_dump(example, all_logits, all_walks_vertices, step=step)
      full_accuracy.append(this_full_accuracy)
    else:
      full_accuracy.append(0)
    if len(full_accuracy) >= total or one_mesh:
      break
  if (step % FLAGS.ev_full_every == 0):
    tf.summary.scalar(name="accuracy_test/full", data=np.mean(full_accuracy), step=step)
  accuracy = acc_per_walk.result()
  evaluate.best_accuracy = max(accuracy, evaluate.best_accuracy)
  tf.summary.scalar(name="accuracy_test/per_walk", data=accuracy, step=step)
  return evaluate.best_accuracy


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

try:
  helpers.setup_tensorboard(FLAGS)
  helpers.summary_command(parser, FLAGS)
  total = tf.data.experimental.cardinality(ds_train).numpy()
  pbar = tqdm.tqdm(ds_train, leave=False, disable=not FLAGS.tqdm)
  for train_example in pbar:
    if not FLAGS.only_evaluate:
      train(train_example)
    best_accuracy = evaluate()
    if FLAGS.only_evaluate:
      print('Eval: ', best_accuracy.numpy())
      exit(0)
    pbar.set_postfix_str("best accuracy: {:.3f}".format(best_accuracy))
    if optimizer.iterations.numpy() % 5000 == 0:
      model.save_weights(FLAGS.logdir, step=optimizer.iterations.numpy(), keep=True)

except KeyboardInterrupt:
  helpers.handle_keyboard_interrupt(FLAGS)
