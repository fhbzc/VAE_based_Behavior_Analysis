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
"""SketchRNN training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time
import urllib
import zipfile

# internal imports

import numpy as np
import six
import tensorflow as tf

import model as sketch_rnn_model


tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'data_dir',
    'https://github.com/hardmaru/sketch-rnn-datasets/raw/master/aaron_sheep',
    'The directory in which to find the dataset specified in model hparams. '
    'If data_dir starts with "http://" or "https://", the file will be fetched '
    'remotely.')
tf.app.flags.DEFINE_string(
    'log_root', 'log/',
    'Directory to store model checkpoints, tensorboard.')
tf.app.flags.DEFINE_boolean(
    'resume_training', False,
    'Set to true to load previous checkpoint')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Pass in comma-separated key=value pairs such as '
    '\'save_every=40,decay_rate=0.99\' '
    '(no whitespace) to be read into the HParams object defined in model.py')


def reset_graph():
  """Closes the current default session and resets the graph."""
  sess = tf.get_default_session()
  if sess:
    sess.close()
  tf.reset_default_graph()



def evaluate_model(sess, model, data_set):
  """Returns the average weighted cost, reconstruction cost and KL cost."""
  total_cost = 0.0
  total_r_cost = 0.0
  total_kl_cost = 0.0
  number_batches = int(np.shape(data_set[0])[0]/model.hps.batch_size)
  for batch in range(number_batches):

    start_idx = batch * model.hps.batch_size
    indices = range(start_idx, start_idx + model.hps.batch_size)
    x, y = get_batch_with_indices(data_set, indices)
    # unused_orig_x, x, s = data_set.get_batch(batch)
    feed = {model.input_data: x, model.output_x: y}
    (cost, r_cost,
     kl_cost) = sess.run([model.cost, model.r_cost, model.kl_cost], feed)
    total_cost += cost
    total_r_cost += r_cost
    total_kl_cost += kl_cost

  total_cost /= (number_batches)
  total_r_cost /= (number_batches)
  total_kl_cost /= (number_batches)
  return (total_cost, total_r_cost, total_kl_cost)


def load_checkpoint(sess, checkpoint_path):
  saver = tf.train.Saver(tf.global_variables())
  ckpt = tf.train.get_checkpoint_state(checkpoint_path)
  tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
  saver.restore(sess, ckpt.model_checkpoint_path)


def save_model(sess, model_save_path, global_step):
  saver = tf.train.Saver(tf.global_variables())
  checkpoint_path = os.path.join(model_save_path, 'vector')
  tf.logging.info('saving model %s.', checkpoint_path)
  tf.logging.info('global_step %i.', global_step)
  saver.save(sess, checkpoint_path, global_step=global_step)

def get_batch_with_indices(dataset,idx):
    input = [] # input
    output = [] # output
    for index in idx:
        input.append(dataset[0][index])
        output.append(dataset[1][index])
    input = np.array(input)
    output = np.array(output)
    return input, output

def train(sess, model, eval_model, train_set, valid_set, test_set):
  """Train a sketch-rnn model."""
  # Setup summary writer.
  summary_writer = tf.summary.FileWriter(FLAGS.log_root)

  # Calculate trainable params.
  t_vars = tf.trainable_variables()
  count_t_vars = 0
  for var in t_vars:
    num_param = np.prod(var.get_shape().as_list())
    count_t_vars += num_param
    tf.logging.info('%s %s %i', var.name, str(var.get_shape()), num_param)
  tf.logging.info('Total trainable variables %i.', count_t_vars)
  model_summ = tf.summary.Summary()
  model_summ.value.add(
      tag='Num_Trainable_Params', simple_value=float(count_t_vars))
  summary_writer.add_summary(model_summ, 0)
  summary_writer.flush()

  # setup eval stats
  best_valid_cost = 100000000.0  # set a large init value
  valid_cost = 0.0

  # main train loop

  hps = model.hps
  start = time.time()

  for _ in range(hps.num_steps):

    step = sess.run(model.global_step)

    curr_learning_rate = ((hps.learning_rate - hps.min_learning_rate) *
                          (hps.decay_rate)**step + hps.min_learning_rate)
    curr_kl_weight = (hps.kl_weight - (hps.kl_weight - hps.kl_weight_start) *
                      (hps.kl_decay_rate)**step)
    # train_set should be [input, output],
    #   input will be of dimension [game_round, time_step, information_dimension]
    #   output will be of dimension [game_round, time_step, 1]
    idx = np.random.permutation(range(0, np.shape(train_set[0])[0]))[0:hps.batch_size]

    x,y = get_batch_with_indices(train_set,idx)

    feed = {
        model.input_data: x,
        model.output_x:y,
        model.lr: curr_learning_rate,
        model.kl_weight: curr_kl_weight
    }

    (train_cost, r_cost, kl_cost, _, train_step, _) = sess.run([
        model.cost, model.r_cost, model.kl_cost, model.final_state,
        model.global_step, model.train_op
    ], feed)

    if step % 20 == 0 and step > 0:

      end = time.time()
      time_taken = end - start

      cost_summ = tf.summary.Summary()
      cost_summ.value.add(tag='Train_Cost', simple_value=float(train_cost))
      reconstr_summ = tf.summary.Summary()
      reconstr_summ.value.add(
          tag='Train_Reconstr_Cost', simple_value=float(r_cost))
      kl_summ = tf.summary.Summary()
      kl_summ.value.add(tag='Train_KL_Cost', simple_value=float(kl_cost))
      lr_summ = tf.summary.Summary()
      lr_summ.value.add(
          tag='Learning_Rate', simple_value=float(curr_learning_rate))
      kl_weight_summ = tf.summary.Summary()
      kl_weight_summ.value.add(
          tag='KL_Weight', simple_value=float(curr_kl_weight))
      time_summ = tf.summary.Summary()
      time_summ.value.add(
          tag='Time_Taken_Train', simple_value=float(time_taken))

      output_format = ('step: %d, lr: %.6f, klw: %0.4f, cost: %.4f, '
                       'recon: %.4f, kl: %.4f, train_time_taken: %.4f')
      output_values = (step, curr_learning_rate, curr_kl_weight, train_cost,
                       r_cost, kl_cost, time_taken)
      output_log = output_format % output_values

      tf.logging.info(output_log)

      summary_writer.add_summary(cost_summ, train_step)
      summary_writer.add_summary(reconstr_summ, train_step)
      summary_writer.add_summary(kl_summ, train_step)
      summary_writer.add_summary(lr_summ, train_step)
      summary_writer.add_summary(kl_weight_summ, train_step)
      summary_writer.add_summary(time_summ, train_step)
      summary_writer.flush()
      start = time.time()

    if step % hps.save_every == 0 and step > 0:

      (valid_cost, valid_r_cost, valid_kl_cost) = evaluate_model(
          sess, eval_model, valid_set)

      end = time.time()
      time_taken_valid = end - start
      start = time.time()

      valid_cost_summ = tf.summary.Summary()
      valid_cost_summ.value.add(
          tag='Valid_Cost', simple_value=float(valid_cost))
      valid_reconstr_summ = tf.summary.Summary()
      valid_reconstr_summ.value.add(
          tag='Valid_Reconstr_Cost', simple_value=float(valid_r_cost))
      valid_kl_summ = tf.summary.Summary()
      valid_kl_summ.value.add(
          tag='Valid_KL_Cost', simple_value=float(valid_kl_cost))
      valid_time_summ = tf.summary.Summary()
      valid_time_summ.value.add(
          tag='Time_Taken_Valid', simple_value=float(time_taken_valid))

      output_format = ('best_valid_cost: %0.4f, valid_cost: %.4f, valid_recon: '
                       '%.4f, valid_kl: %.4f, valid_time_taken: %.4f')
      output_values = (min(best_valid_cost, valid_cost), valid_cost,
                       valid_r_cost, valid_kl_cost, time_taken_valid)
      output_log = output_format % output_values

      tf.logging.info(output_log)

      summary_writer.add_summary(valid_cost_summ, train_step)
      summary_writer.add_summary(valid_reconstr_summ, train_step)
      summary_writer.add_summary(valid_kl_summ, train_step)
      summary_writer.add_summary(valid_time_summ, train_step)
      summary_writer.flush()

      if valid_cost < best_valid_cost:
        best_valid_cost = valid_cost

        save_model(sess, FLAGS.log_root, step)

        end = time.time()
        time_taken_save = end - start
        start = time.time()

        tf.logging.info('time_taken_save %4.4f.', time_taken_save)

        best_valid_cost_summ = tf.summary.Summary()
        best_valid_cost_summ.value.add(
            tag='Best_Valid_Cost', simple_value=float(best_valid_cost))

        summary_writer.add_summary(best_valid_cost_summ, train_step)
        summary_writer.flush()

        (eval_cost, eval_r_cost, eval_kl_cost) = evaluate_model(
            sess, eval_model, test_set)

        end = time.time()
        time_taken_eval = end - start
        start = time.time()

        eval_cost_summ = tf.summary.Summary()
        eval_cost_summ.value.add(tag='Eval_Cost', simple_value=float(eval_cost))
        eval_reconstr_summ = tf.summary.Summary()
        eval_reconstr_summ.value.add(
            tag='Eval_Reconstr_Cost', simple_value=float(eval_r_cost))
        eval_kl_summ = tf.summary.Summary()
        eval_kl_summ.value.add(
            tag='Eval_KL_Cost', simple_value=float(eval_kl_cost))
        eval_time_summ = tf.summary.Summary()
        eval_time_summ.value.add(
            tag='Time_Taken_Eval', simple_value=float(time_taken_eval))

        output_format = ('eval_cost: %.4f, eval_recon: %.4f, '
                         'eval_kl: %.4f, eval_time_taken: %.4f')
        output_values = (eval_cost, eval_r_cost, eval_kl_cost, time_taken_eval)
        output_log = output_format % output_values

        tf.logging.info(output_log)

        summary_writer.add_summary(eval_cost_summ, train_step)
        summary_writer.add_summary(eval_reconstr_summ, train_step)
        summary_writer.add_summary(eval_kl_summ, train_step)
        summary_writer.add_summary(eval_time_summ, train_step)
        summary_writer.flush()


def trainer(model_params):
  """Train a sketch-rnn model."""
  np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

  tf.logging.info('sketch-rnn')
  tf.logging.info('Hyperparams:')
  for key, val in six.iteritems(model_params.values()):
    tf.logging.info('%s = %s', key, str(val))
  tf.logging.info('Loading data files.')


  _ = np.load(model_params.train_data_set, encoding="latin1")
  train_set = {}
  train_set[0] = _["input"].copy()
  train_set[1] = _["output"].copy()

  _ = np.load(model_params.valid_data_set, encoding="latin1")
  valid_set = {}
  valid_set[0] = _["input"].copy()
  valid_set[1] = _["output"].copy()

  _ = np.load(model_params.test_data_set, encoding="latin1")
  test_set = {}
  test_set[0] = _["input"].copy()
  test_set[1] = _["output"].copy()


  eval_model_params = sketch_rnn_model.copy_hparams(model_params)


  reset_graph()
  model = sketch_rnn_model.Model(model_params)
  eval_model = sketch_rnn_model.Model(eval_model_params, reuse=True)

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  if FLAGS.resume_training:
    load_checkpoint(sess, FLAGS.log_root)

  # Write config file to json file.
  tf.gfile.MakeDirs(FLAGS.log_root)
  with tf.gfile.Open(
      os.path.join(FLAGS.log_root, 'model_config.json'), 'w') as f:
    json.dump(model_params.values(), f, indent=True)

  train(sess, model, eval_model, train_set, valid_set, test_set)


def main(unused_argv):
  """Load model params, save config file and start trainer."""
  model_params = sketch_rnn_model.get_default_hparams()
  # All the paramaters have to be defined in model.py, get_default_hparams()
  # if FLAGS.hparams:
  #   model_params.parse(FLAGS.hparams)
  trainer(model_params)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
