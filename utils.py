import logging
import numpy as np
import tensorflow as tf
from matplotlib import gridspec, pyplot as plt

# main.py
def mod_name(prev_name, n_epochs, is_pretrain, snr=None, lr=None, beta1=None, beta2=None, gain=None):
    
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
        mod_split.append('lr({})_betas({},{})_gain({})_epoch({})'.format(lr, beta1, beta2, gain, n_epochs))
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

