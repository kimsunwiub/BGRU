import re
import logging
import numpy as np
import tensorflow as tf
from matplotlib import gridspec, pyplot as plt

# main.py
def mod_name(prev_name, n_epochs, is_pretrain, snr=None, lr=None, beta1=None, beta2=None, gain=None, clip_val=None, d1=None, d2=None, d3=None):
    
    def mod_n_epochs(sp, n_epochs):
        prev_n_epochs = int(re.split(r'[()]', sp)[1])
        sp = 'epoch({})'.format(prev_n_epochs + n_epochs)
        return sp
    
    def mod_sp(sp, n_epochs, snr):
        if 'epoch' in sp:
            sp = mod_n_epochs(sp, n_epochs)
        elif snr and 'SNR' in sp:
            sp = 'SNR({:.4f})'.format(snr)
        return sp
    
    mod_split = []
    is_first = True
    
    if is_pretrain:
        for sp in prev_name.split('_'):
            if 'epoch' in sp:
                is_first = False
            sp = mod_sp(sp, n_epochs, snr)
            mod_split.append(sp)
            
    else:
        prev_split = prev_name.split('pretrain')
        
        if len(prev_split) == 2:
            prev_split[-1] += '_'
            prev_split.append('(False)')
        else:
            is_first = False
            temp = []
            for sp in prev_split[-1].split('_'):
                sp = mod_sp(sp, n_epochs, snr)
                temp.append(sp)
                
            prev_split[-1] = '_'.join(temp)

        mod_split = 'pretrain'.join(prev_split).split('_')
        
    if is_first:
        mod_split.append('lr({})_betas({},{})_gain({})_clip({})_dropouts({},{},{})_epoch({})'.format(lr, beta1, beta2, gain, clip_val, d1, d2, d3, n_epochs))
        if snr: 
            mod_split.append('SNR({:.4f})'.format(snr))
        
    return '_'.join(mod_split)

def plot_results(model, fn):
    plt.switch_backend('agg')
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1])
    plt.subplot(gs[0]); plt.plot(model.tr_losses); plt.title('Tr_Losses')
    plt.subplot(gs[1]); plt.plot(model.va_losses); plt.title('Va_Losses')
    plt.subplot(gs[2]); plt.plot(model.va_snrs); plt.title('Va_SNRs')
    plt.tight_layout()
    plt.savefig('%s.png' % fn)
    logging.info('Saving results to {}'.format(fn))


@tf.RegisterGradient("SigmoidGrads")
def sigmoid_grad(op, grad):
    x, = op.inputs
    return grad * x * (1.0 - x) * 2

@tf.RegisterGradient("TanhGrads")
def tanh_grad(op, grad):
    x, = op.inputs
    return grad * (1-tf.pow(tf.tanh(x), 2))

"""
def B_sigmoid(x):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign":"SigmoidGrads"}):
        return 0.5 * (tf.sign(x)+1)

def B_tanh(x):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign":"TanhGrads"}):
        return tf.sign(x)
"""

@tf.custom_gradient
def B_sigmoid(x):
    def grad(dy):
        return dy * tf.sigmoid(x) * (1.0 - tf.sigmoid(x)) * 2
    return tf.round(0.5 * (tf.sign(x)+1)), grad

@tf.custom_gradient
def B_tanh(x):
    def grad(dy):
        return dy * (1-tf.pow(tf.tanh(x), 2))
    return tf.sign(x), grad

# Models.py
def empty_array(size):
    """
    Initialize an array of 'size' number of nans
    
    Params:
        size: Size of the array
    """
    return np.full(size, np.nan)

def saver_dict(trainable_variables):
    names = ['W_out', 'b_out',
        #'init_0', 'init_1',
        'gk0','gb0', 'ck0', 'cb0',
        'gk1','gb1', 'ck1', 'cb1']
    saver_dict = {}
    for n,t in zip(names, trainable_variables):
        saver_dict[n] = t
    return saver_dict

def compute_SNR(S, S_hat):
    """
    Helper function to retrieve SNR
    
    Params:
        S: orignal source
        S_hat: recovered
    """
    numerator = np.sum(np.power(S,2), axis=1)
    denominator = np.sum(np.power(S - S_hat, 2), axis=1)
    return np.mean(10*np.log10(numerator/denominator))

# Consumes more memory -->
def matmul_sparse_outputs(outputs, w_):
    sparse_outputs = tf.contrib.layers.dense_to_sparse(outputs)
    return tf.sparse_tensor_dense_matmul(sparse_outputs, w_)

def matmul_sparse_weights(outputs, w_):
    w_T = tf.transpose(w_)
    sparse_w_T = tf.contrib.layers.dense_to_sparse(w_T)
    o_T = tf.transpose(outputs)
    result = tf.sparse_tensor_dense_matmul(sparse_w_T, o_T)
    return tf.transpose(result)
# <--

def get_mask(weight, rho=0.95):
    shape = weight.get_shape().as_list()
    allshape = shape[0]
    if len(shape) > 1:
        allshape = shape[0] * shape[1]
    
    # Find threshold
    W_ = tf.reshape(weight, [-1])
    th_p = int(np.round(allshape*rho))
    sorted_ = tf.gather(W_, tf.nn.top_k(W_, k=allshape).indices)
    th = tf.gather(sorted_, th_p)
    
    # Create mask
    mask_less = tf.less(weight,th)
    mask_greater = tf.greater(weight,-th)
    mask = tf.logical_or(mask_less, mask_greater)
    return mask

def normalize(tensor):
    return tf.div(tensor, tf.reduce_max(tensor))

def variable_summaries(var, name_scope):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name_scope):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
  
