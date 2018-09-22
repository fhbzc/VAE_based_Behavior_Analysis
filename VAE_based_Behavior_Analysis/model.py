# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sketch-RNN Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# internal imports

import numpy as np
import tensorflow as tf

import rnn


def copy_hparams(hparams):
  """Return a copy of an HParams instance."""
  return tf.contrib.training.HParams(**hparams.values())


def get_default_hparams():
  """Return default HParams for sketch-rnn."""
  hparams = tf.contrib.training.HParams(
      train_data_set="train_dataset.npz",  # Our dataset.
      valid_data_set="valid_dataset.npz",
      test_data_set="test_dataset.npz",
      num_steps=10000000,  # Total number of steps of training. Keep large.
      save_every=500,  # Number of batches per checkpoint creation.
      max_seq_len=20,  # Not used. Will be changed by model. [Eliminate?]
      dec_rnn_size=1,  # Size of decoder.
      dec_model='lstm',  # Decoder: lstm, layer_norm or hyper.
      enc_rnn_size=128,  # Size of encoder.
      enc_model='lstm',  # Encoder: lstm, layer_norm or hyper.
      z_size=32,  # Size of latent vector z. Recommend 32, 64 or 128.
      kl_weight=0.5,  # KL weight of loss equation. Recommend 0.5 or 1.0.
      kl_weight_start=0.01,  # KL start weight when annealing.
      kl_tolerance=0.02,  # Level of KL loss at which to stop optimizing for KL.
      batch_size=3,  # Minibatch size. Recommend leaving at 100.
      grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
      learning_rate=0.01,  # Learning rate.
      decay_rate=0.9999,  # Learning rate decay per minibatch.
      kl_decay_rate=0.99995,  # KL annealing decay rate per minibatch.
      min_learning_rate=0.00000001,  # Minimum learning rate.
      use_recurrent_dropout=True,  # Dropout with memory loss. Recomended
      recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep.
      use_input_dropout=False,  # Input dropout. Recommend leaving False.
      input_dropout_prob=0.90,  # Probability of input dropout keep.
      use_output_dropout=False,  # Output droput. Recommend leaving False.
      output_dropout_prob=0.90,  # Probability of output dropout keep.
      random_scale_factor=0.15,  # Random scaling data augmention proportion.
      augment_stroke_prob=0.10,  # Point dropping augmentation proportion.
      conditional=True,  # When False, use unconditional decoder-only model.
      is_training=True,  # Is model training? Recommend keeping true.
      input_seq_len = 10,
      input_dimension_real = 4,
      input_dimension_get = 7,
  )
  return hparams


class Model(object):
  """Define a SketchRNN model."""

  def __init__(self, hps, gpu_mode=True, reuse=False):
    """Initializer for the SketchRNN model.

    Args:
       hps: a HParams object containing model hyperparameters
       gpu_mode: a boolean that when True, uses GPU mode.
       reuse: a boolean that when true, attemps to reuse variables.
    """
    self.hps = hps
    with tf.variable_scope('vector_rnn', reuse=reuse):
      if not gpu_mode:
        with tf.device('/cpu:0'):
          tf.logging.info('Model using cpu.')
          self.build_model(hps)
      else:
        tf.logging.info('Model using gpu.')
        self.build_model(hps)

  def encoder(self, batch):
    """Define the bi-directional encoder module of sketch-rnn."""
    unused_outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
        self.enc_cell_fw,
        self.enc_cell_bw,
        batch,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='ENC_RNN')

    last_state_fw, last_state_bw = last_states
    last_h_fw = self.enc_cell_fw.get_output(last_state_fw)
    last_h_bw = self.enc_cell_bw.get_output(last_state_bw)
    last_h = tf.concat([last_h_fw, last_h_bw], 1)
    mu = rnn.super_linear(
        last_h,
        self.hps.z_size,
        input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
        scope='ENC_RNN_mu',
        init_w='gaussian',
        weight_start=0.001)
    # presig = rnn.super_linear(
    #     last_h,
    #     self.hps.z_size,
    #     input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
    #     scope='ENC_RNN_sigma',
    #     init_w='gaussian',
    #     weight_start=0.001)
    return mu

  def build_model(self, hps):
    """Define model architecture."""
    if hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    if hps.dec_model == 'lstm':
      cell_fn = rnn.LSTMCell
    elif hps.dec_model == 'layer_norm':
      cell_fn = rnn.LayerNormLSTMCell
    elif hps.dec_model == 'hyper':
      cell_fn = rnn.HyperLSTMCell
    else:
      assert False, 'please choose a respectable cell'

    if hps.enc_model == 'lstm':
      enc_cell_fn = rnn.LSTMCell
    elif hps.enc_model == 'layer_norm':
      enc_cell_fn = rnn.LayerNormLSTMCell
    elif hps.enc_model == 'hyper':
      enc_cell_fn = rnn.HyperLSTMCell
    else:
      assert False, 'please choose a respectable cell'

    use_recurrent_dropout = self.hps.use_recurrent_dropout
    use_input_dropout = self.hps.use_input_dropout
    use_output_dropout = self.hps.use_output_dropout

    cell = cell_fn(
        hps.dec_rnn_size,
        use_recurrent_dropout=use_recurrent_dropout,
        dropout_keep_prob=self.hps.recurrent_dropout_prob)

    if hps.conditional:  # vae mode:
        self.enc_cell_fw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
        self.enc_cell_bw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)

    # dropout:
    tf.logging.info('Input dropout mode = %s.', use_input_dropout)
    tf.logging.info('Output dropout mode = %s.', use_output_dropout)
    tf.logging.info('Recurrent dropout mode = %s.', use_recurrent_dropout)
    if use_input_dropout:
      tf.logging.info('Dropout to input w/ keep_prob = %4.4f.',
                      self.hps.input_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, input_keep_prob=self.hps.input_dropout_prob)
    if use_output_dropout:
      tf.logging.info('Dropout to output w/ keep_prob = %4.4f.',
                      self.hps.output_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=self.hps.output_dropout_prob)
    self.cell = cell


    self.input_data = tf.placeholder(
        dtype=tf.float32,
        shape=[self.hps.batch_size, self.hps.max_seq_len, self.hps.input_dimension_get])
    self.input_handle = self.input_data[:,:,:self.hps.input_dimension_real]
    # The target/expected vectors of strokes
    self.output_x = tf.placeholder(
        dtype=tf.float32,
        shape=[self.hps.batch_size, self.hps.max_seq_len, 1]) # always 0 or 1 indicates one's action

    # vectors of strokes to be fed to decoder (same as above, but lagged behind
    # one step to include initial dummy value of (0, 0, 1, 0, 0))
    self.input_x = self.input_handle

    # either do vae-bit and get z, or do unconditional, decoder-only
    if hps.conditional:  # vae mode:
      _ = tf.concat([self.input_handle[:,:self.hps.input_seq_len,:],self.output_x[:,:self.hps.input_seq_len,:]],axis=2)
      self.mean = self.encoder(_)
      # self.sigma = tf.exp(self.presig / 2.0)  # sigma > 0. div 2.0 -> sqrt.
      # eps = tf.random_normal(
      #     (self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)
      self.batch_z = self.mean


      # KL cost
      self.kl_cost = -0.5 * tf.reduce_mean(
          (1 + - tf.square(self.mean) ))
      self.kl_cost = tf.maximum(self.kl_cost, self.hps.kl_tolerance)
      pre_tile_y = tf.reshape(self.batch_z,
                              [self.hps.batch_size, 1, self.hps.z_size])
      overlay_x = tf.tile(pre_tile_y, [1, self.hps.max_seq_len, 1])
      actual_input_x = tf.concat([self.input_x, overlay_x], 2)
      self.initial_state = tf.nn.tanh(
          rnn.super_linear(
              self.batch_z,
              cell.state_size,
              init_w='gaussian',
              weight_start=0.001,
              input_size=self.hps.z_size))
    else:  # unconditional, decoder-only generation
      self.batch_z = tf.zeros(
          (self.hps.batch_size, self.hps.z_size), dtype=tf.float32)
      self.kl_cost = tf.zeros([], dtype=tf.float32)
      actual_input_x = self.input_x
      self.initial_state = cell.zero_state(
          batch_size=hps.batch_size, dtype=tf.float32)


    # TODO(deck): Better understand this comment.
    # Number of outputs is 3 (one logit per pen state) plus 6 per mixture
    # component: mean_x, stdev_x, mean_y, stdev_y, correlation_xy, and the
    # mixture weight/probability (Pi_k)
    # n_out = (3 + self.num_mixture * 6)
    n_out = 1
    with tf.variable_scope('RNN'):
      output_w = tf.get_variable('output_w', [self.hps.dec_rnn_size, n_out])
      output_b = tf.get_variable('output_b', [n_out])

    # decoder module of sketch-rnn is below
    output, last_state = tf.nn.dynamic_rnn(
        cell,
        actual_input_x,
        initial_state=self.initial_state,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='RNN')

    output = tf.reshape(output, [-1, hps.dec_rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    self.final_state = last_state

    label = tf.reshape(self.output_x, [self.hps.batch_size, self.hps.max_seq_len])
    out = tf.reshape(output, [self.hps.batch_size, self.hps.max_seq_len])
    self.sigmoid_out = tf.sigmoid(out)
    # self.r_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out[:,self.hps.input_seq_len:], labels=label[:,self.hps.input_seq_len:]))
    self.r_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=label))
    if self.hps.is_training:
      self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
      optimizer = tf.train.AdamOptimizer(self.lr)

      self.kl_weight = tf.Variable(self.hps.kl_weight_start, trainable=False)
      self.cost = self.r_cost + self.kl_cost * self.kl_weight

      gvs = optimizer.compute_gradients(self.cost)
      g = self.hps.grad_clip
      capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
      self.train_op = optimizer.apply_gradients(
          capped_gvs, global_step=self.global_step, name='train_step')
