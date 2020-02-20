import os 
from argparse import ArgumentParser
import numpy as np
import _pickle as pickle
from matplotlib import gridspec, pyplot as plt
import librosa
import librosa.display
import time
from tqdm import tqdm

import tensorflow as tf
#from Initializers import my_xavier_initializer

from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.layers import base as base_layer
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell

def SDR(s,sr):
    eps=1e-20
    ml=np.minimum(len(s), len(sr))
    s=s[:ml]
    sr=sr[:ml]
    return ml, 10*np.log10(np.sum(s**2)/(np.sum((s-sr)**2)+eps)+eps)

def saver_dict_p1(trainable_variables):
    names = ['W_out', 'b_out',
        'gk','gb', 'ck', 'cb']
    saver_dict = {}
    for n,t in zip(names, trainable_variables):
        saver_dict[n] = t
    return saver_dict

def get_mean_mask(temp_w, _t, _th):
    orig_shape = temp_w.get_shape().as_list()
    
    wv = tf.reshape(temp_w, [-1])
    wv_len = wv.get_shape().as_list()[0]
    th_idx = int(wv_len*(1-_t))
    abswv = tf.abs(wv)

    _topk = tf.nn.top_k(abswv, k=th_idx)
    sorted_ = tf.gather(abswv, _topk.indices)
    if th_idx == 0: _mu = tf.zeros(1)
    else: _mu = tf.reduce_mean(sorted_)
        
    s_mask_ = tf.zeros(orig_shape, dtype=tf.float32)
    s_mask_p = tf.where(temp_w > _th, tf.ones(orig_shape) * _mu, s_mask_)
    s_mask_np = tf.where(temp_w < -_th, tf.ones(orig_shape) * _mu, s_mask_p)
    return s_mask_np

def get_sparsity_threshold(temp_w, _t):
    wv = tf.reshape(temp_w, [-1])
    wv_len = wv.get_shape().as_list()[0]
    abswv = tf.abs(wv)

    _topk = tf.nn.top_k(abswv, k=wv_len)

    sorted_ = tf.gather(abswv, _topk.indices)

    th_idx = int(wv_len*(1-_t))
    if th_idx == wv_len: _th = sorted_[th_idx-1]
    else: _th = sorted_[th_idx]
    return _th

def masking_step(w, _t, percinc):
    _th = get_sparsity_threshold(tf.tanh(w), _t)
    sparmask = get_mean_mask(tf.tanh(w), _t, _th)
    _shape = w.get_shape().as_list()
    
    # Non-intensive (Edit on 9/28)
    bernoulli = tf.distributions.Bernoulli(probs=percinc, dtype=tf.float32)
    percmask = bernoulli.sample(_shape)
    
    w_masked = B_tanh(w)*sparmask*percmask + tf.tanh(w)*(1-percmask)
    return w_masked

@tf.custom_gradient
def B_sigmoid(x):
    def grad(dy):
        return dy * tf.sigmoid(x) * (1.0 - tf.sigmoid(x)) * 2
    return tf.round(0.5 * (tf.sign(x)+1)), grad

@tf.custom_gradient
def B_tanh(x):
    def grad(dy):
        return dy * (1-tf.pow(tf.tanh(x), 2))
    return tf.sign(x+1e-9), grad

class StochasticPercincBinaryGRUCell(tf.contrib.rnn.LayerRNNCell):

  def __init__(self,
               num_units,
               W_gate,
               b_gate,
               W_cand,
               b_cand,
               
               # Non-intensive (Edit on 9/28)
               #g_percmask,
               #c_percmask,
               percinc,
               
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None):
    super(StochasticPercincBinaryGRUCell , self).__init__(_reuse=reuse, name=name)

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
    self.percinc = percinc
    
    
    # Non-intensive (Edit on 9/28)
    #self.g_percmask = g_percmask
    #self.c_percmask = c_percmask

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
    
    # Non-intensive (Edit on 9/28)
    bernoulli = tf.distributions.Bernoulli(probs=self.percinc, dtype=tf.float32)
    
    # Gate
    inputs_gi = array_ops.concat([inputs, state], 1)
    gate_inputs = math_ops.matmul(inputs_gi, self._gate_kernel) + self._gate_bias
                          
    # Non-intensive (Edit on 9/28)
    _shape = gate_inputs.get_shape().as_list()
    percmask = bernoulli.sample(_shape)
    value = B_sigmoid(gate_inputs)*percmask + tf.sigmoid(gate_inputs)*(1-percmask)
    
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
    r_state = r * state

    # Cand
    inputs_cand = array_ops.concat([inputs, r_state], 1)
    candidate = math_ops.matmul(inputs_cand, self._candidate_kernel) + self._candidate_bias
    
    # Non-intensive (Edit on 9/28)
    _shape = candidate.get_shape().as_list()
    percmask = bernoulli.sample(_shape)
    c = B_tanh(candidate)*percmask + tf.tanh(candidate)*(1-percmask)

    new_h = (1 - u) * state + u * c
    return new_h, new_h

def parse_arguments():
    parser = ArgumentParser()
    
    parser.add_argument("node_id", type=int,
                        help="Node ID")
    parser.add_argument("gpu_id", type=int,
                        help="GPU ID")
    parser.add_argument("learning_rate", type=float, 
                        help="Learning rate")
    parser.add_argument("percinc", type=float,
                        help="Binarization percentage")
    parser.add_argument("prev_model", type=str,
                        help="Load previous checkpoint model")
    
    parser.add_argument("--bptt", type=int, default=50,
                        help="BPTT Value")
    parser.add_argument("_t", type=float, default=0.8,
                        help="Sparsity rate")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Number of epochs")
    parser.add_argument("--beta1", type=float, default=0.4,
                        help="Beta1 value for AdamOptimizer")
    parser.add_argument("--beta2", type=float, default=0.9,
                        help="Beta2 value for AdamOptimizer")
    parser.add_argument("-j", "--dropout1", type=float, default=0.05,
                        help="Dropout value for inputs (Default: 0.05)")
    parser.add_argument("-l", "--dropout2", type=float, default=0.20,
                        help="Dropout value for outputs (Default: 0.20)")
    parser.add_argument("-m", "--dropout_cell", type=float, default=0.20,
                        help="Dropout value for GRU Cell (Default: 0.20)")
    parser.add_argument("--save_model", type=int, default=1,
                        help="Decide Saving Weights. (1: True, Else False)")
    parser.add_argument("-s", "--bs", type=int, default=10,
                        help="Batch Size (Default: 10)")
    
    return parser.parse_args()

args = parse_arguments()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
_n = args.num_epochs
data = pickle.load(open("data_bgru.pkl", "rb"))

trXH, trY = data['trXH'], data['trY']
teX, teXH, teY, teS = data['teX'], data['teXH'], data['teY'], data['teS']

H=1000
D_in = 513*4
D_out = 513
bs = 10

len_trXH = 1200 # Number of training signals
len_teX = 400 # Number of test signals

save_nm = "Phase_2_Ep{}_Perc{}".format(args.num_epochs, args.percinc)

tf.compat.v1.reset_default_graph()

inputs = tf.compat.v1.placeholder(tf.float32, [bs, args.bptt, D_in], name='inputs_placeholder')
target = tf.compat.v1.placeholder(tf.float32, [bs, args.bptt, D_out], name='target_placeholder')

# Initializers
# initializer=tf.contrib.layers.xavier_initializer(uniform=False)
initializer=tf.keras.initializers.GlorotNormal()

with tf.compat.v1.variable_scope("out_layer"):
    W_out = tf.compat.v1.get_variable("W_out", [H, D_out], initializer=initializer)
    b_out = tf.compat.v1.get_variable("b_out", [D_out], initializer=initializer)
with tf.compat.v1.variable_scope("layer_1"):
    gk0 = tf.compat.v1.get_variable("gk", [D_in+H, H*2], initializer=initializer)
    gb0 = tf.compat.v1.get_variable("gb", [H*2], initializer=initializer)
    ck0 = tf.compat.v1.get_variable("ck", [D_in+H, H], initializer=initializer)
    cb0 = tf.compat.v1.get_variable("cb", [H], initializer=initializer)

gk0_masked = masking_step(gk0, args._t, args.percinc)
gb0_masked = masking_step(gb0, args._t, args.percinc)
ck0_masked = masking_step(ck0, args._t, args.percinc)
cb0_masked = masking_step(cb0, args._t, args.percinc)

# GRU Cell
GRUcell = StochasticPercincBinaryGRUCell(H, gk0_masked, gb0_masked, ck0_masked, cb0_masked, args.percinc)
tf.reset_default_graph(); 

inputs = tf.placeholder(tf.float32, [bs, args.bptt, D_in], name='inputs_placeholder')
target = tf.placeholder(tf.float32, [bs, args.bptt, D_out], name='target_placeholder')

with tf.variable_scope("out_layer"):
    W_out = tf.get_variable("W_out", [H, D_out])
    b_out = tf.get_variable("b_out", [D_out])
with tf.variable_scope("layer_1"):
    gk0 = tf.get_variable("gk", [D_in+H, H*2])
    gb0 = tf.get_variable("gb", [H*2])
    ck0 = tf.get_variable("ck", [D_in+H, H])
    cb0 = tf.get_variable("cb", [H])

gk0_masked = masking_step(gk0, args._t, args.percinc)
gb0_masked = masking_step(gb0, args._t, args.percinc)
ck0_masked = masking_step(ck0, args._t, args.percinc)
cb0_masked = masking_step(cb0, args._t, args.percinc)

GRUcell = StochasticPercincBinaryGRUCell(H, gk0_masked, gb0_masked, ck0_masked, cb0_masked, args.percinc)

states_series, current_state = tf.nn.dynamic_rnn(GRUcell, inputs, dtype=tf.float32)
outputs = tf.reshape(states_series, [-1, H])

W_out_masked = masking_step(W_out, args._t, args.percinc)
b_out_masked = masking_step(b_out, args._t, args.percinc)

logits_inputs = tf.matmul(outputs, W_out_masked) + b_out_masked

# Non-intensive (Edit on 9/28)
bernoulli = tf.distributions.Bernoulli(probs=args.percinc, dtype=tf.float32)
_shape = logits_inputs.get_shape().as_list()
percmask = bernoulli.sample(_shape)
logits = B_tanh(logits_inputs)*percmask + tf.tanh(logits_inputs)*(1-percmask)
trYh = tf.reshape(logits, [bs, -1, D_out])

# Training loss
tr_err = tf.reduce_sum(tf.square(tf.subtract(trYh, target)))

# Optimizer
opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2)
gvs = opt.compute_gradients(tr_err)
op = opt.apply_gradients(gvs)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

with tf.compat.v1.Session() as sess:
    tic = time.time()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver(saver_dict_p1(tf.compat.v1.trainable_variables()))
    saver.restore(sess, 'Saved_Models/{}'.format(args.prev_model))
    for e in range(args.num_epochs):
        curr_tr_err,curr_te_err,curr_te_snr = [],[],[]
        # Train
        for i in range(0, len(trXH), args.bs):
        #for i in range(0, len(trXH)//10, args.bs):
            feed_x_batch = np.array(trXH[i:i+args.bs])
            feed_y_batch = np.array(trY[i:i+args.bs])
            for j in range(0, feed_x_batch.shape[-1], args.bptt):
                feed_x_bptt = np.transpose(feed_x_batch[:,:,j:j+args.bptt], (0,2,1))
                feed_y_bptt = np.transpose(feed_y_batch[:,:,j:j+args.bptt], (0,2,1))
                if feed_x_bptt.shape[1] >= args.bptt:
                    _tr_err, _ = sess.run([tr_err, op], 
                                          feed_dict={inputs: feed_x_bptt, target: feed_y_bptt})
                    curr_tr_err.append(_tr_err)

        mean_tr_err = np.array(curr_tr_err).mean()
        tot_tr_err.append(mean_tr_err)
        print ("Epoch {} Tr_Err {}".format(e, mean_tr_err))
        #Test
        for i in range(0, len(teXH), args.bs):
            feed_x_batch = np.array(teX[i:i+args.bs])
            feed_xh_batch = np.array(teXH[i:i+args.bs])
            feed_y_batch = np.array(teY[i:i+args.bs])
            feed_s_batch = np.array(teS[i:i+args.bs])
            for j in range(0, feed_x_batch.shape[-1], args.bptt):
                feed_x_bptt = np.transpose(feed_x_batch[:,:,j:j+args.bptt], (0,2,1))
                feed_xh_bptt = np.transpose(feed_xh_batch[:,:,j:j+args.bptt], (0,2,1))
                feed_y_bptt = np.transpose(feed_y_batch[:,:,j:j+args.bptt], (0,2,1))
                feed_s_bptt = feed_s_batch[:,:,j:j+args.bptt]
                if feed_x_bptt.shape[1] >= args.bptt:
                    _te_err, pred_y, _ = sess.run([tr_err, trYh, op], 
                                                  feed_dict={inputs: feed_xh_bptt, target: feed_y_bptt})
                    curr_te_err.append(_te_err)
                    s_pred = np.transpose(feed_x_bptt * np.round((pred_y+1)*.5), (0,2,1))
                    for k in range(bs):
                        the_snr = SDR(librosa.istft(s_pred[k]), librosa.istft(feed_s_bptt[k]))[1]
                        curr_te_snr.append(the_snr)
                        #print ("k: {}, shat: {}, s: {}, snr: {}".format(k, s_pred[k].shape, feed_s_bptt[k].shape, the_snr))
        mean_te_err = np.array(curr_te_err).mean()
        mean_te_snr = np.array(curr_te_snr).mean()
        tot_te_err.append(mean_te_err)
        tot_te_snr.append(mean_te_snr)
        print ("Epoch {} Te_Err {} Te_SDR {}".format(e, mean_te_err, mean_te_snr))
        
    print('Saving as {}. Avg SNR: {:.4f}'.format(save_nm, np.array(tot_te_snr).mean()))
    print ("args.save_model: {}".format(args.save_model))
    saver.save(sess, 'Saved_Models/{}'.format(save_nm))