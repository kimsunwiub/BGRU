from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.layers import base as base_layer
import tensorflow as tf
from utils import get_mask, B_tanh, B_sigmoid

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell

@tf_export("nn.rnn_cell.GRUCell") #? Whats this
class TanhGRUCell(LayerRNNCell):

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None):
    super(TanhGRUCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    self._gate_kernel = self.add_variable(
        "gates/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_bias = self.add_variable(
        "gates/%s" % _BIAS_VARIABLE_NAME,
        shape=[2 * self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.constant_initializer(1.0, dtype=self.dtype)))
    self._candidate_kernel = self.add_variable(
        "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_bias = self.add_variable(
        "candidate/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    
    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), tf.tanh(self._gate_kernel))
    gate_inputs = nn_ops.bias_add(gate_inputs, tf.tanh(self._gate_bias))

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    
    candidate = math_ops.matmul(
        array_ops.concat([inputs, r_state], 1), tf.tanh(self._candidate_kernel))
    candidate = nn_ops.bias_add(candidate, tf.tanh(self._candidate_bias))

    c = self._activation(candidate)
    new_h = (1 - u) * state + u * c
    return new_h, new_h

@tf_export("nn.rnn_cell.GRUCell")
class BinaryGRUCell(LayerRNNCell):

  def __init__(self,
               num_units,
               W_gate,
               b_gate,
               W_cand,
               b_cand,
               rho,
               activation=None,
               reuse=None):
    super(BinaryGRUCell, self).__init__(_reuse=reuse, name=None)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = None
    self._bias_initializer = None
    self._gate_kernel = W_gate
    self._gate_bias = b_gate
    self._candidate_kernel = W_cand
    self._candidate_bias = b_cand
    self.rho = rho

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    mask_w, mask_b = get_mask(self._gate_kernel, self.rho), get_mask(self._gate_bias, self.rho)
    w_ = tf.where(mask_w, B_tanh(self._gate_kernel), tf.zeros(self._gate_kernel.shape))
    b_ = tf.where(mask_b, B_tanh(self._gate_bias), tf.zeros(self._gate_bias.shape))
    
    gate_inputs = tf.matmul(array_ops.concat([inputs, state], 1), w_)
    gate_inputs = nn_ops.bias_add(gate_inputs, b_)

    value = B_sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    mask_w, mask_b = get_mask(self._candidate_kernel, self.rho), get_mask(self._candidate_bias, self.rho)
    w_ = tf.where(mask_w, B_tanh(self._candidate_kernel), tf.zeros(self._candidate_kernel.shape))
    b_ = tf.where(mask_b, B_tanh(self._candidate_bias), tf.zeros(self._candidate_bias.shape))
    
    candidate = tf.matmul(array_ops.concat([inputs, r_state], 1), w_)
    candidate = nn_ops.bias_add(candidate, b_)

    c = B_tanh(candidate)
    new_h = (1 - u) * state + u * c
    return new_h, new_h

class ScalingTanhGRUCell(tf.contrib.rnn.LayerRNNCell):

  def __init__(self,
               num_units,
               W_gate,
               b_gate,
               W_cand,
               b_cand,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None):
    super(ScalingTanhGRUCell , self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = None
    self._bias_initializer = None
    self._gate_kernel = W_gate
    self._gate_bias = b_gate
    self._candidate_kernel = W_cand
    self._candidate_bias = b_cand

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    
    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), self._gate_kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    value = B_sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    w_ = self._candidate_kernel
    b_ = self._candidate_bias
    candidate = math_ops.matmul(
        array_ops.concat([inputs, r_state], 1), w_)
    candidate = nn_ops.bias_add(candidate, b_)

    c = B_tanh(candidate)
    new_h = (1 - u) * state + u * c
    return new_h, new_h