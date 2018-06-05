from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
import tensorflow as tf

_WEIGHTS_VARIABLE_NAME = "kernel"
_BIAS_VARIABLE_NAME = "bias"

def binary_with_sigmoid_grad(x):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign":"SigmoidGrads"}):
        return 0.5 * (tf.sign(x)+1)

def bipolar_binarize_with_tanh_grad(x):
    # <Change name>
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign":"TanhGrads"}):
        return tf.sign(x)

class TanhGRUCell(tf.nn.rnn_cell.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(TanhGRUCell, self).__init__(_reuse=reuse)
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

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope("gates"):  # Reset gate and update gate.
      # We start with bias of 1.0 to not reset and not update.
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        dtype = [a.dtype for a in [inputs, state]][0]
        bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
      value = math_ops.sigmoid(
          _linear([inputs, state], 2 * self._num_units, True, bias_ones,
                  self._kernel_initializer))
      r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
    with vs.variable_scope("candidate"):
      c = self._activation(
          _linear([inputs, r * state], self._num_units, True,
                  self._bias_initializer, self._kernel_initializer))
    new_h = u * state + (1 - u) * c
    return new_h, new_h

class BinaryGRUCell(tf.nn.rnn_cell.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self,
               num_units,
               W_gate,
               b_gate,
               W_cand,
               b_cand,
               activation=None,
               reuse=False):
    super(BinaryGRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = None
    self._bias_initializer = None
    
    self.W_gate = W_gate
    self.b_gate = b_gate
    self.W_cand = W_cand
    self.b_cand = b_cand
    
  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope("gates"):  # Reset gate and update gate.
      value = binary_with_sigmoid_grad(
              _binary_linear([inputs, state], 2 * self._num_units, 
                  self.W_gate, self.b_gate))
      r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
      
    with vs.variable_scope("candidate"):
      c = bipolar_binarize_with_tanh_grad(
          _binary_linear([inputs, r * state], self._num_units,
                  self.W_cand, self.b_cand))
    new_h = u * state + (1 - u) * c
    return new_h, new_h



def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """    
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]
  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), math_ops.tanh(weights))
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, math_ops.tanh(biases))


def _binary_linear(args,
            output_size,
            weights,
            bias):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable."""
  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    total_arg_size += shape[1].value
  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    res = math_ops.matmul(array_ops.concat(args, 1), bipolar_binarize_with_tanh_grad(weights))
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
    return nn_ops.bias_add(res, bipolar_binarize_with_tanh_grad(bias))