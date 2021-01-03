from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
layers = tf.keras.layers


class RnnWalkNet(tf.keras.Model):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn=None,
               dump_model_visualization=True,
               optimizer=None):
    super(RnnWalkNet, self).__init__(name='RnnWalkNet')

    self._layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512}

    self._classes = classes
    self._params = params

    self._init_layers()
    inputs = tf.keras.layers.Input(shape=(100, net_input_dim))
    self.build(input_shape=(1, 100, net_input_dim))
    outputs = self.call(inputs)
    if dump_model_visualization:
      tmp_model = keras.Model(inputs=inputs, outputs=outputs, name='WalkModel')
      tmp_model.summary(print_fn=self._print_fn)
      tf.keras.utils.plot_model(tmp_model, params.logdir + '/RnnWalkModel.png', show_shapes=True)

    self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
    self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self._params.logdir, max_to_keep=5)

  def _print_fn(self, st):
    with open(self._params.logdir + '/log.txt', 'at') as f:
      f.write(st + '\n')

  def load_weights(self, filepath):
    super(RnnWalkNet, self).load_weights(filepath)

  def save_weights(self, folder, step=None, keep=False):
    self.manager.save()
    if keep:
      super(RnnWalkNet, self).save_weights(folder + '/learned_model2keep__' + str(step).zfill(8) + '.keras')

  def _init_layers(self):
    kernel_regularizer = tf.keras.regularizers.l2(0.0001)
    initializer = tf.initializers.Orthogonal(3)
    self._norm1 = tfa.layers.InstanceNormalization(axis=2)
    self._norm2 = tfa.layers.InstanceNormalization(axis=2)
    self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    rnn_layer = layers.GRU
    self._gru1 = rnn_layer(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    self._gru2 = rnn_layer(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    self._gru3 = rnn_layer(self._layer_sizes['gru3'], time_major=False,
                           return_sequences=not self._params.one_label_per_model,
                           return_state=False,
                           recurrent_initializer=initializer, kernel_initializer=initializer,
                           kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                           bias_regularizer=kernel_regularizer)
    self._fc_last = layers.Dense(self._classes, activation='softmax', kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)


  def call(self, model_ftrs, training=True):
    x = model_ftrs[:, 1:]
    x = self._fc1(x)
    x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    x = self._fc2(x)
    x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    x1 = self._gru1(x, training=training)
    x2 = self._gru2(x1, training=training)
    x3 = self._gru3(x2, training=training)
    x = x3

    x = self._fc_last(x)

    return x

