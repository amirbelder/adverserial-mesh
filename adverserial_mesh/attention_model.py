from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict
import copy

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import utils

from tensorflow import keras
layers = tf.keras.layers

''' https://www.tensorflow.org/tutorials/text/transformer  '''

def get_windowed_mask(inp_size, window_n):
    '''
    :param inp_size: 3D size input: batches, seq_len, features
    :param window_n: size of window for mask - how many neighbors each vertex will consider (N must be odd)
    :return:
    '''
    diag_mat = tf.linalg.diag(inp_size[1])
    each_side = int((window_n -1) / 2)
    # TODO: finish this one
    return



def dense_layer(size, activation=None, use_bias=True,
                initializer=tf.initializers.Orthogonal(1), regulizer=tf.keras.regularizers.l2(0.0001)):
    return tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias, kernel_initializer=initializer,
                                 kernel_regularizer=regulizer, bias_regularizer=regulizer)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def coordinate_encoding(position, max_time_step, d_model):
    pos_encoding = positional_encoding(position, d_model)
    step_signal = positional_encoding(max_time_step, d_model)
    return pos_encoding, step_signal


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      dense_layer(dff, activation='relu'),  # (batch_size, seq_len, dff)
      dense_layer(d_model)  # (batch_size, seq_len, d_model)
  ])


def scaled_dot_product_attention(q, k, v, mask):
  qk = tf.matmul(q, k, transpose_b=True)
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention = qk / tf.math.sqrt(dk)
  if mask is not None:
    scaled_attention += (mask * -1e9)
  weights = tf.nn.softmax(scaled_attention, axis=-1)
  output = tf.matmul(weights, v)
  return output, weights


class CrossWalkAttention(keras.layers.Layer):
  def __init__(self, num_neurons, num_heads, out_dim, n_walks_per_model):
    super(CrossWalkAttention, self).__init__()

    self.n_walks = n_walks_per_model
    self.num_heads = num_heads
    self.num_neurons = num_neurons
    self.depth = num_neurons // self.num_heads
    self.attention_layer = scaled_dot_product_attention

    self.q_layer = dense_layer(num_neurons, initializer=None)
    self.k_layer = dense_layer(num_neurons, initializer=None)
    self.v_layer = dense_layer(num_neurons, initializer=None)

    self.linear_layer = dense_layer(out_dim, initializer=None)

  def split(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, x, mask):
    # TODO: currently gets a single weight for each walk, change to weight matrix of N_walks x Features_dim
    xs = x.shape
    x = tf.reshape(x, (-1, self.n_walks, xs[-1]))
    batch_size = tf.shape(x)[0]

    # Run through linear layers
    q = self.q_layer(x)
    k = self.k_layer(x)
    v = self.v_layer(x)

    # Split the heads
    q = self.split(q, batch_size)
    k = self.split(k, batch_size)
    v = self.split(v, batch_size)

    # Run through attention
    attention_output, weights = self.attention_layer(q, k, v, mask)

    # Prepare for the rest of processing
    output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    concat_attention = tf.reshape(output, (batch_size, -1, self.num_neurons))

    output = tf.reduce_sum(tf.nn.softmax(concat_attention, axis=1)* x, axis=1)
    return output, tf.nn.softmax(concat_attention, axis=1)


class MultiHeadAttention(keras.layers.Layer):
  def __init__(self, num_neurons, num_heads):
    super(MultiHeadAttention, self).__init__()

    self.num_heads = num_heads
    self.num_neurons = num_neurons
    self.depth = num_neurons // self.num_heads
    self.attention_layer = scaled_dot_product_attention

    self.q_layer = dense_layer(num_neurons, regulizer=None)
    self.k_layer = dense_layer(num_neurons, regulizer=None)
    self.v_layer = dense_layer(num_neurons, regulizer=None)

    self.linear_layer = dense_layer(num_neurons)

  def split(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    # Run through linear layers
    q = self.q_layer(q)
    k = self.k_layer(k)
    v = self.v_layer(v)

    # Split the heads
    q = self.split(q, batch_size)
    k = self.split(k, batch_size)
    v = self.split(v, batch_size)

    # Run through attention
    attention_output, weights = self.attention_layer(q, k, v, mask)

    # Prepare for the rest of processing
    output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    concat_attention = tf.reshape(output, (batch_size, -1, self.num_neurons))

    # Run through final linear layer
    output = self.linear_layer(concat_attention)

    return output, weights



class LocalGlobalAttention(keras.layers.Layer):
    def __init__(self, num_neurons, num_heads, local_window_len):
        super(LocalGlobalAttention, self).__init__()
        self.local_attention_layer = MultiHeadAttention(num_neurons, num_heads)
        self.global_attention_layer = MultiHeadAttention(num_neurons, num_heads)
        self.local_window_len = local_window_len

    def call(self, x, k, q, mask=None):
        # TODO: reshape according to local_window_len and attending each short walk as its own
        xs = x.shape
        x_global_attn, global_weights = self.global_attention_layer(x, k, q, None)
        x = tf.reshape(x, (-1, self.local_window_len, xs[-1]))
        x_local_attn, local_weights = self.local_attention_layer(x, k, q, None)
        x_local_attn = tf.reshape(x_local_attn, (xs[0], -1, xs[-1]))
        # x = tf.reshape(x, ())
        return x_global_attn + x_local_attn, tf.stack([global_weights, local_weights])



class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, window_len=None):
        super(EncoderLayer, self).__init__()
        if window_len is None:
            self.mha = MultiHeadAttention(d_model, num_heads)
        else:
            self.mha = LocalGlobalAttention(d_model, num_heads, window_len)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.layernorm1 = tfa.layers.InstanceNormalization(axis=-1, epsilon=1e-7)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.layernorm2 = tfa.layers.InstanceNormalization(axis=-1, epsilon=1e-7)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        self.gate = tf.keras.layers.Dense(d_model, kernel_initializer=tf.initializers.Identity(),
                                          use_bias=False, activation='sigmoid')

    def call(self, x, training, mask):
        attn_output, attn_map = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        # TODO: change this line for GATED self attention, let x + G(attn_output) decide how much to take from attention
        gated = self.gate(x)
        out1 = self.layernorm1 (gated* x + (1.0 - gated) * attn_output, training=training)  # (batch_size, input_seq_len, d_model)
        # out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output, training=training)  # (batch_size, input_seq_len, d_model)

        return out2, attn_map


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-7)
        # self.layernorm1 = tfa.layers.InstanceNormalization(axis=-1, epsilon=1e-7)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-7)
        # self.layernorm2 = tfa.layers.InstanceNormalization(axis=-1, epsilon=1e-7)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-7)
        # self.layernorm3 = tfa.layers.InstanceNormalization(axis=-1, epsilon=1e-7)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class FC_embedder(tf.keras.layers.Layer):
    def __init__(self, fc1_size, fc2_size):
        super(FC_embedder, self).__init__()
        k_reg = tf.keras.regularizers.l2(0.0001)
        initializer = tf.initializers.Orthogonal(1)
        self.fc1 = layers.Dense(fc1_size,  kernel_regularizer=k_reg, bias_regularizer=k_reg,
                             kernel_initializer=initializer)
        # self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm1 = tfa.layers.InstanceNormalization(axis=-1)
        self.fc2 = layers.Dense(fc2_size, kernel_regularizer=k_reg, bias_regularizer=k_reg,
                                kernel_initializer=initializer)
        # self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tfa.layers.InstanceNormalization(axis=-1)

    def call(self, x, training):
        x = self.fc1(x)
        x = self.norm1(x, training=training)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = self.norm2(x, training=training)
        x = tf.nn.relu(x)
        return x


class FaceEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Dense(64)

        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        x = self.embedding(x)


class MHA_embedder(tf.keras.layers.Layer):
    def __init__(self, nn1, nn2, num_heads, k):
        super(MHA_embedder, self).__init__()
        # self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm1 = tfa.layers.InstanceNormalization(axis=2)
        # self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tfa.layers.InstanceNormalization(axis=2)
        self.mha1 = MultiHeadAttention(nn1, num_heads)
        self.mha2 = MultiHeadAttention(nn2, num_heads)
        self.k = k

    @tf.function
    def call(self, x, training):
        xs = tf.shape(x)
        x = tf.reshape(x, (-1, self.k, xs[-1]))
        x, _ = self.mha1(x, x, x, None)
        x = self.norm1(x, training=training)
        x = tf.nn.relu(x)
        x, _ = self.mha2(x, x, x, None)
        x = self.norm2(x, training=training)
        x = tf.nn.relu(x)
        x = tf.reshape(x, (xs[0], xs[1], x.shape[-1]))
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = FC_embedder(d_model//2, d_model)
        # self.embedding = MHA_embedder(128, d_model, 1, maximum_position_encoding)


        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, maximum_position_encoding)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x, training=training)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers-1):
            x, attn_map = self.enc_layers[i](x, training, mask)
            x += self.pos_encoding[:, :seq_len, :]

        x, attn_map = self.enc_layers[-1](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)


class RecurrentEncoder(tf.keras.layers.Layer):
    def __init__(self, num_timesteps, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.0):
        super(RecurrentEncoder, self).__init__()

        self.d_model = d_model
        self.n_ts = num_timesteps

        self.embedding = FC_embedder(d_model//2, d_model)
        # self.embedding = MHA_embedder(128, d_model, 1, maximum_position_encoding)

        self.pos_encoding, self.time_encoding = coordinate_encoding(maximum_position_encoding, num_timesteps,
                                                self.d_model)

        self.recurrent_layer = EncoderLayer(d_model, num_heads, dff, rate)


        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x, training=training)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        for i in range(self.n_ts-1):
            x += self.pos_encoding[:, :seq_len, :]
            x += tf.expand_dims(self.time_encoding[:, i, :], axis=1)
            x = self.dropout(x, training=training)
            x, attn_map = self.recurrent_layer(x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def     __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, K,
                 maximum_position_encoding, rate=0.1, pooling=False):
        super(Decoder, self).__init__()

        self.pooling = pooling
        self.d_model = d_model
        self.num_layers = num_layers
        self.K = K

        # self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.embedding = FC_embedder(d_model//2, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        if self.pooling:
            x = tf.reshape(tf.reduce_mean(tf.reshape(x, (-1, self.K, x.shape[-1])), axis=1),
                                (-1, x.shape[-2] // self.K, x.shape[-1]))
        seq_len = tf.shape(x)[1]

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, params, rate=0.1,
                 model_must_be_load=False,
                 optimizer=None,
                 model_fn=None):
        super(Transformer, self).__init__()
        # Legacy params - needed for train_val iterations
        self._model_must_be_load = model_must_be_load
        self._classes = params.n_classes


        self.params = params
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        # self.final_layer = tf.keras.layers.Dense(target_vocab_size)

        self._fc_last = layers.Dense(self.params.n_classes, activation=self.params.last_layer_actication,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                     bias_regularizer=tf.keras.regularizers.l2(0.0001),
                                     kernel_initializer=tf.initializers.Orthogonal(3))

        if optimizer:
            if model_fn:
                # self.checkpoint = tf.train.Checkpoint(optimizer=copy.deepcopy(optimizer), model=self)
                self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
            else:
                self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
            self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self.params.logdir, max_to_keep=2)
            if model_fn:  # Transfer learning
                self.load_weights(model_fn)
                self.checkpoint.optimizer = optimizer
            else:
                self.load_weights()
        else:
            self.checkpoint = tf.train.Checkpoint(model=self)
            if model_fn:
                self.load_weights(model_fn)
            # else:
            #     self.load_weights(tf.train.latest_checkpoint(self.params.logdir))

    def load_weights(self, filepath=None):
        if filepath is not None and filepath.endswith('.keras'):
            super(Transformer, self).load_weights(filepath)
        elif filepath is None:
            status = self.checkpoint.restore(self.manager.latest_checkpoint)
            print(utils.color.BLUE, 'Starting from iteration: ', self.checkpoint.optimizer.iterations.numpy(),
                  utils.color.END)
        else:
            filepath = filepath.replace('//', '/')
            status = self.checkpoint.restore(filepath)

    def save_weights(self, folder, step=None, keep=False):
        if self.manager is not None:
            self.manager.save()
        if keep:
            super(Transformer, self).save_weights(folder + '/learned_model2keep__' + str(step).zfill(8) + '.keras')
            # self.checkpoint.write(folder + '/learned_model2keep--' + str(step))

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask, both=False):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        # final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        final_output = self._fc_last(dec_output)
        if both:
            return attention_weights, final_output[:,-1,:]
        else:
            return attention_weights, final_output


class WalkTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 out_features, pe_input, pe_target, params, rate=0.25,
                 model_must_be_load=False,
                 optimizer=None,
                 model_fn=None,
                 num_scales=None):
        super(WalkTransformer, self).__init__()
        # Legacy params - needed for train_val iterations
        self._model_must_be_load = model_must_be_load
        self._classes = params.n_classes


        self.params = params
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)


        self.final_layer = tf.keras.layers.Dense(out_features, activation=self.params.last_layer_actication,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                     bias_regularizer=tf.keras.regularizers.l2(0.0001),
                                     kernel_initializer=tf.initializers.Orthogonal(3))

        self._fc_last = layers.Dense(self.params.n_classes, activation=self.params.last_layer_actication,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                     bias_regularizer=tf.keras.regularizers.l2(0.0001),
                                     kernel_initializer=tf.initializers.Orthogonal(3))


        s_in = (200, self.params.net_input_dim)
        build_s_in = (8, 4,) + s_in
        inputs = tf.keras.layers.Input(shape=s_in)
        self.build(input_shape=build_s_in)

        if optimizer:
            if model_fn:
                # self.checkpoint = tf.train.Checkpoint(optimizer=copy.deepcopy(optimizer), model=self)
                self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
            else:
                self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
            self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self.params.logdir, max_to_keep=2)
            if model_fn:  # Transfer learning
                self.load_weights(model_fn)
                self.checkpoint.optimizer = optimizer
            else:
                self.load_weights()
        else:
            self.checkpoint = tf.train.Checkpoint(model=self)
            if model_fn:
                self.load_weights(model_fn)
            else:
                self.load_weights(tf.train.latest_checkpoint(self.params.logdir))

    def load_weights(self, filepath=None):
        if filepath is not None and filepath.endswith('.keras'):
            super(WalkTransformer, self).load_weights(filepath)
        elif filepath is None:
            status = self.checkpoint.restore(self.manager.latest_checkpoint)
            print(utils.color.BLUE, 'Starting from iteration: ', self.checkpoint.optimizer.iterations.numpy(),
                  utils.color.END)
        else:
            filepath = filepath.replace('//', '/')
            status = self.checkpoint.restore(filepath)

    def save_weights(self, folder, step=None, keep=False):
        if self.manager is not None:
            self.manager.save()
        if keep:
            super(WalkTransformer, self).save_weights(folder + '/learned_model2keep__' + str(step).zfill(8) + '.keras')
            # self.checkpoint.write(folder + '/learned_model2keep--' + str(step))

    # def call(self, inp, tar, training, enc_padding_mask,
    #          look_ahead_mask, dec_padding_mask, both=False):
    def call(self, inp, enc_padding_mask=None, training=True, both=False, classify=True):
        xs = inp.shape
        inp = tf.reshape(inp, (-1, xs[-2], xs[-1]))  # reshaping to [batch*n_per_shape, walk_len, features_dimension]
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        final_features = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)

        final_output = self._fc_last(final_features)
        if both:
            # return attention_weights, final_output[:,-1,:]
            return final_output, final_output[:, -1, :]
        elif classify:
            # return attention_weights, final_output
            return tf.reduce_mean(final_output, axis=1)
        else:
            return final_features


class WalkHierTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 out_features, pe_input, pe_target, params, rate=0.0,
                 jump_every_k=10,
                 pooling= False,
                 concat_xyz=False,
                 model_must_be_load=False,
                 optimizer=None,
                 model_fn=None,
                 num_scales=None,
                 global_dim_mult=1,
                 recurrent=False):
        super(WalkHierTransformer, self).__init__()
        # Legacy params - needed for train_val iterations
        self._model_must_be_load = model_must_be_load
        self._classes = params.n_classes
        self.cross_walk_attn = params.cross_walk_attn if hasattr(params, 'cross_walk_attn') else False
        self.dropout = tf.keras.layers.Dropout(rate)
        self.params = params
        self.K = jump_every_k
        self.pooling = pooling if self.params.one_label_per_model else False
        self.concat_xyz = concat_xyz
        if recurrent:
            self.local_encoder = RecurrentEncoder(num_layers, d_model, num_heads, dff,
                                     input_vocab_size, jump_every_k, rate)
        else:
            self.local_encoder = Encoder(num_layers, d_model, num_heads, dff,
                                         input_vocab_size, jump_every_k, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, jump_every_k, self.params.seq_len, rate, self.pooling)
        # self.global_layer1 = EncoderLayer(d_model * global_dim_mult, num_heads, dff * global_dim_mult, rate=rate)
        # TODO: change global layer to GRU? GRU + attention?
        self._fc_last = layers.Dense(self.params.n_classes, activation=self.params.last_layer_actication
                                     )
        # kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        #                              bias_regularizer=tf.keras.regularizers.l2(0.0001),
        #                              kernel_initializer=tf.initializers.Orthogonal(3))
        if self.cross_walk_attn:
            self._cw_attn_layer = CrossWalkAttention(d_model, 1, 1, self.params.n_walks_per_model)


        s_in = (params.seq_len, self.params.net_input_dim)
        # build_s_in = (8, 4,) + s_in
        build_s_in = (self.params.n_walks_per_model * 4,) + s_in
        inputs = tf.keras.layers.Input(shape=s_in)
        self.build(input_shape=build_s_in)

        if optimizer:
            if model_fn:
                # self.checkpoint = tf.train.Checkpoint(optimizer=copy.deepcopy(optimizer), model=self)
                self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
            else:
                self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
            self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self.params.logdir, max_to_keep=1)
            if model_fn:  # Transfer learning
                self.load_weights(model_fn)
                self.checkpoint.optimizer = optimizer
            else:
                self.load_weights()
        else:
            self.checkpoint = tf.train.Checkpoint(model=self)
            if model_fn:
                self.load_weights(model_fn)
            else:
                self.load_weights(tf.train.latest_checkpoint(self.params.logdir))

    def load_weights(self, filepath=None):
        if filepath is not None and filepath.endswith('.keras'):
            super(WalkHierTransformer, self).load_weights(filepath)
        elif filepath is None:
            status = self.checkpoint.restore(self.manager.latest_checkpoint)
            print(utils.color.BLUE, 'Starting from iteration: ', self.checkpoint.optimizer.iterations.numpy(),
                  utils.color.END)
        else:
            filepath = filepath.replace('//', '/')
            status = self.checkpoint.restore(filepath)

    def save_weights(self, folder, step=None, keep=False):
        if self.manager is not None:
            self.manager.save()
        if keep:
            super(WalkHierTransformer, self).save_weights(folder + '/learned_model2keep__' + str(step).zfill(8) + '.keras')
            # self.checkpoint.write(folder + '/learned_model2keep--' + str(step))

    # def call(self, inp, tar, training, enc_padding_mask,
    #          look_ahead_mask, dec_padding_mask, both=False):

    def single_hier_block(self, block_layer, x, k, training):
        xs = x.shape
        x = tf.reshape(x, (-1, k, xs[-1]))
        enc_output = block_layer(x, training, None)
        if self.pooling:
            enc_output = tf.reduce_mean(enc_output, axis=1)
            rs_size = xs[-2] // k
        else:
            rs_size = xs[-2]
        return tf.reshape(enc_output, (-1, rs_size, enc_output.shape[-1]))


    @tf.function
    def call(self, inp, enc_padding_mask=None, training=True, both=False, classify=True):
        enc_output = self.single_hier_block(self.local_encoder, inp, self.K, training)
        # enc_output, attn_map = self.global_layer1(enc_output, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(
            inp, enc_output, training, None, None)
        # attn_list = tf.nn.softmax(tf.reduce_sum(tf.reduce_sum(attn_map, axis=-2), axis=-2), axis=-1)
        if self.params.one_label_per_model:
            dec_output = tf.reduce_mean(dec_output, axis=1)
        if self.cross_walk_attn:
            dec_output, weights = self._cw_attn_layer(dec_output, None)
        final_output = self._fc_last(dec_output)
        classification = final_output
        if classify== 'both':
            # return attention_weights, final_output[:,-1,:]
            # return final_output, tf.reduce_sum(final_output * tf.expand_dims(attn_list, axis=-1), axis=1)
            return classification, dec_output
        elif classify == 'weights':
            return classification, weights, dec_output
        elif classify:
            # return attention_weights, final_output
            return classification   # before - was slicing[:, -1, :]
        else:
            return weights, final_output, tf.reduce_mean(final_output, axis=1)



if __name__ == '__main__':
    params = EasyDict()
    params.last_layer_actication = 'softmax'
    params.n_classes=40
    params.logdir = None

    sample_transformer = Transformer(
        num_layers=2, d_model=512, num_heads=8, dff=2048,
        input_vocab_size=8500, target_vocab_size=256,
        pe_input=10000, pe_target=6000, params=params)

    temp_input = tf.random.uniform((8, 400, 7))
    temp_target = tf.random.uniform((8, 400, 7)) # TODO: think if this is even relevant. we dont have a sequence output, or do we? semantic segmentation?

    fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

    # OUTPUT IS CORRECT SIZE, NEED TO TEST THIS MUMBO JUMBO NONSENSE