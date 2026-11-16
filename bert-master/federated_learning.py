# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Federated learning implementation for BERT-based models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import os
import re
import numpy as np
import six
# TensorFlow 2.x 兼容
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import modeling
import optimization
import tokenization

def create_federated_model_fn(bert_config, num_labels, init_checkpoint, learning_rate,
                            num_train_steps, num_warmup_steps, use_tpu,
                            use_one_hot_embeddings):
  """Returns `model_fn` closure for federated learning."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for federated learning."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = modeling.create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn

def get_model_parameters(estimator):
  """Get model parameters from estimator."""
  variables = estimator.get_variable_names()
  params = {}
  for var in variables:
    params[var] = estimator.get_variable_value(var)
  return params

def set_model_parameters(estimator, params):
  """Set model parameters for estimator."""
  for var_name, var_value in params.items():
    estimator.set_variable_value(var_name, var_value)

def federated_averaging(params_list, weights=None):
  """Federated averaging of model parameters.
  
  Args:
    params_list: List of model parameters from different clients.
    weights: Optional list of weights for each client.
    
  Returns:
    Aggregated model parameters.
  """
  if not params_list:
    return None
  
  if weights is None:
    weights = [1.0 / len(params_list)] * len(params_list)
  
  aggregated_params = {}
  for var_name in params_list[0].keys():
    var_values = [params[var_name] * w for params, w in zip(params_list, weights)]
    aggregated_params[var_name] = sum(var_values)
  
  return aggregated_params

def secure_aggregation(params_list, secure_aggregation_config=None):
  """Secure aggregation of model parameters.
  
  Args:
    params_list: List of model parameters from different clients.
    secure_aggregation_config: Configuration for secure aggregation.
    
  Returns:
    Aggregated model parameters.
  """
  # This is a simplified version of secure aggregation
  # In practice, this would involve encryption and secure multi-party computation
  return federated_averaging(params_list)

def federated_train_and_aggregate(trainers, num_rounds=10):
  """Train model federatively and aggregate parameters.
  
  Args:
    trainers: List of client trainers.
    num_rounds: Number of federated learning rounds.
    
  Returns:
    Aggregated model parameters.
  """
  for round_num in range(num_rounds):
    tf.logging.info("Starting federated learning round %d", round_num + 1)
    
    # Train on each client
    client_params = []
    for i, trainer in enumerate(trainers):
      tf.logging.info("Training on client %d", i + 1)
      trainer.train()
      params = trainer.get_parameters()
      client_params.append(params)
    
    # Aggregate parameters
    aggregated_params = federated_averaging(client_params)
    
    # Distribute aggregated parameters back to clients
    for trainer in trainers:
      trainer.set_parameters(aggregated_params)
    
  return aggregated_params

class FederatedClientTrainer:
  """Federated learning client trainer."""
  
  def __init__(self, model_fn, train_input_fn, eval_input_fn, model_dir):
    """Initialize federated client trainer.
    
    Args:
      model_fn: Model function.
      train_input_fn: Training input function.
      eval_input_fn: Evaluation input function.
      model_dir: Model directory.
    """
    self.model_fn = model_fn
    self.train_input_fn = train_input_fn
    self.eval_input_fn = eval_input_fn
    self.model_dir = model_dir
    
    self.estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir)
  
  def train(self, max_steps=100):
    """Train model on client data.
    
    Args:
      max_steps: Maximum number of training steps.
    """
    self.estimator.train(
        input_fn=self.train_input_fn,
        max_steps=max_steps)
  
  def evaluate(self):
    """Evaluate model on client data.
    
    Returns:
      Evaluation metrics.
    """
    return self.estimator.evaluate(
        input_fn=self.eval_input_fn)
  
  def get_parameters(self):
    """Get model parameters.
    
    Returns:
      Model parameters.
    """
    return get_model_parameters(self.estimator)
  
  def set_parameters(self, params):
    """Set model parameters.
    
    Args:
      params: Model parameters.
    """
    set_model_parameters(self.estimator, params)

class FederatedServer:
  """Federated learning server."""
  
  def __init__(self, model_fn, model_dir):
    """Initialize federated server.
    
    Args:
      model_fn: Model function.
      model_dir: Model directory.
    """
    self.model_fn = model_fn
    self.model_dir = model_dir
    
    self.estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir)
  
  def aggregate(self, client_params_list):
    """Aggregate client parameters.
    
    Args:
      client_params_list: List of client parameters.
      
    Returns:
      Aggregated parameters.
    """
    return federated_averaging(client_params_list)
  
  def broadcast(self, params):
    """Broadcast parameters to clients.
    
    Args:
      params: Parameters to broadcast.
    """
    # In practice, this would involve sending parameters to clients
    return params

class HierarchicalFederatedServer:
  """Hierarchical federated learning server."""
  
  def __init__(self, model_fn, model_dir, num_levels=2):
    """Initialize hierarchical federated server.
    
    Args:
      model_fn: Model function.
      model_dir: Model directory.
      num_levels: Number of hierarchy levels.
    """
    self.model_fn = model_fn
    self.model_dir = model_dir
    self.num_levels = num_levels
    
    self.servers = []
    for i in range(num_levels):
      level_dir = os.path.join(model_dir, "level_%d" % i)
      server = FederatedServer(model_fn, level_dir)
      self.servers.append(server)
  
  def hierarchical_aggregate(self, client_params_list):
    """Hierarchical aggregation of client parameters.
    
    Args:
      client_params_list: List of client parameters.
      
    Returns:
      Aggregated parameters.
    """
    # Level 0: Aggregate client parameters
    level_0_params = self.servers[0].aggregate(client_params_list)
    
    # Higher levels: Aggregate from lower levels
    for i in range(1, self.num_levels):
      # In practice, this would involve aggregation from multiple lower-level servers
      level_i_params = self.servers[i].aggregate([level_0_params])
      level_0_params = level_i_params
    
    return level_0_params
