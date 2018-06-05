from GRU_Modifications import TanhGRUCell, BinaryGRUCell
from tensorflow.python.ops import math_ops
from librosa.core import istft as istft
import _pickle as pickle
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import re

def mod_name(prev_name, n_epochs, is_pretrain, snr=None, lr=None):
    
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
        mod_split.append('lr({})_epoch({})'.format(lr, n_epochs))
        if snr: 
            mod_split.append('SNR({:.4f})'.format(snr))
        
    return '_'.join(mod_split)

def empty_array(size):
    """
    Initialize an array of 'size' number of nans
    
    Params:
        size: Size of the array
    """
    return np.full(size, np.nan)

def saver_dict(trainable_variables):
    names = ['W_out', 'b_out',
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
    
class GRU_Net(object):

    def __init__(self, perc, bptt, n_epochs, learning_rate, batch_sz, feat, n_layers, state_sz, verbose, is_restore, model_nm, n_bits, is_pretrain):
        """
        feat: Number of features / classes
        """
        self.perc = perc
        self.bptt = bptt
        self.feat = feat
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.batch_sz = batch_sz
        self.state_sz = state_sz
        self.learning_rate = learning_rate
        self.model_nm = model_nm
        self.is_restore = is_restore
        self.n_bits = n_bits
        self.is_pretrain = is_pretrain
        
        if verbose: tf.logging.set_verbosity(tf.logging.DEBUG)
        else: tf.logging.set_verbosity(tf.logging.INFO)
        
        tf.reset_default_graph()
        self.build_inputs()
        self.build_GRU()
        self.build_loss()
        self.build_optimizer()
    
    def build_inputs(self):
        if self.n_bits:
            self.inputs = tf.placeholder(tf.float32, 
                            [None, None, self.feat * self.n_bits]) 
        else:
            self.inputs = tf.placeholder(tf.float32, 
                            [None, None, self.feat]) 
        self.targets = tf.placeholder(tf.float32, 
                        [None, None, self.feat])

    def build_GRU(self):
        
        if self.is_pretrain:
            multicell = tf.contrib.rnn.MultiRNNCell(
                [TanhGRUCell(self.state_sz) for _ in range(self.n_layers)]
                )
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            W_out = tf.Variable(initializer([self.state_sz, self.feat]))
            b_out = tf.Variable(initializer([self.feat]))
            
        else:    
            bW_out = tf.get_variable("W_out", [self.state_sz, self.feat])
            bb_out = tf.get_variable("b_out", [self.feat])
            
            gk0 = tf.get_variable("gk0", [self.feat*self.n_bits+self.state_sz, self.state_sz*2])
            gb0 = tf.get_variable("gb0", [self.state_sz*2])
            ck0 = tf.get_variable("ck0", [self.feat*self.n_bits+self.state_sz, self.state_sz])
            cb0 = tf.get_variable("cb0", [self.state_sz])

            gk1 = tf.get_variable("gk1", [self.state_sz*2, self.state_sz*2])
            gb1 = tf.get_variable("gb1", [self.state_sz*2])
            ck1 = tf.get_variable("ck1", [self.state_sz*2, self.state_sz])
            cb1 = tf.get_variable("cb1", [self.state_sz])

            tanh_gk0 = math_ops.tanh(gk0)
            tanh_gb0 = math_ops.tanh(gb0)
            tanh_ck0 = math_ops.tanh(ck0)
            tanh_cb0 = math_ops.tanh(cb0)

            tanh_gk1 = math_ops.tanh(gk1)
            tanh_gb1 = math_ops.tanh(gb1)
            tanh_ck1 = math_ops.tanh(ck1)
            tanh_cb1 = math_ops.tanh(cb1)
            
            cells = []
            cell = BinaryGRUCell(self.state_sz, tanh_gk0, tanh_gb0, tanh_ck0, tanh_cb0); cells.append(cell)
            cell = BinaryGRUCell(self.state_sz, tanh_gk1, tanh_gb1, tanh_ck1, tanh_cb1); cells.append(cell)
            multicell = tf.contrib.rnn.MultiRNNCell(cells)
            W_out = math_ops.tanh(bW_out)
            b_out = math_ops.tanh(bb_out)
            
        states_series, current_state = tf.nn.dynamic_rnn(multicell, self.inputs, dtype=tf.float32)
        outputs = tf.reshape(states_series,[-1,self.state_sz])

        if self.is_pretrain:
            self.logits = tf.sigmoid(tf.matmul(outputs, tf.tanh(W_out)) + tf.tanh(b_out))
            self.logits = tf.reshape(self.logits, [self.batch_sz, -1, self.feat])
        else:
            def binary_with_sigmoid_grad(x):
                g = tf.get_default_graph()
                with g.gradient_override_map({"Sign":"SigmoidGrads"}):
                    return 0.5 * (tf.sign(x)+1)
            
            def bipolar_binarize_with_tanh_grad(x):
                # <Change name>
                g = tf.get_default_graph()
                with g.gradient_override_map({"Sign":"TanhGrads"}):
                    return tf.sign(x)
            
            self.logits = tf.sigmoid(tf.matmul(outputs, 
                bipolar_binarize_with_tanh_grad(W_out)) + bipolar_binarize_with_tanh_grad(b_out))
            self.logits = tf.reshape(self.logits, [self.batch_sz, -1, self.feat])
        
    def build_loss(self):
        logits_1d = tf.reshape(self.logits, [-1])
        labels = tf.reshape(self.targets, [-1])
        self.loss = tf.reduce_mean(tf.square(tf.subtract(logits_1d, labels)))
        
    def build_optimizer(self):
        self.op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    
    def train(self, data):
        
        def _train(data_tr):
            """
            Helper function to train(): Training
            """
            # Initialize result arrays
            n_iter = len(data_tr['M'])//self.batch_sz
            signal_losses = empty_array(n_iter)

            # Iterate for n_iter (N/b) times
            for i in range(n_iter):
                # Batch inputs
                start_idx = i*self.batch_sz
                end_idx = (i+1)*self.batch_sz
                if self.n_bits:
                    X = np.array(data_tr['M_bin'][start_idx:end_idx], dtype=np.float32)
                else:
                    X = np.array(data_tr['M'][start_idx:end_idx]).real.astype(np.float32)
                y = np.array(data_tr['y'][start_idx:end_idx], dtype=np.float32)
                
                # Run model w.r.t. lookback length
                if self.bptt == -1: local_bptt = X[0].shape[0]
                else: local_bptt = self.bptt
                assert (local_bptt <= X[0].shape[0])
                
                feed_sz = X[0].shape[0]//local_bptt
                feed_losses = empty_array(feed_sz)
                
                for j in range(feed_sz):
                    start_idx = j*local_bptt
                    end_idx = (j+1)*local_bptt
                    X_feed = X[:,start_idx:end_idx]
                    y_feed = y[:,start_idx:end_idx]
                    
                    _, loss = sess.run(
                        [self.op, self.loss],
                        feed_dict={
                            self.inputs: X_feed,
                            self.targets: y_feed
                        })
                    feed_losses[j] = loss
                
                signal_losses[i] = feed_losses.mean()
            return signal_losses.mean()
        
        def _validate(data_va):
            """
            Helper function to train(): Validation
            """
            # Initialize result arrays
            n_iter = len(data_va['M'])//self.batch_sz
            signal_losses, signal_snrs = empty_array((2,n_iter))
            
            # Iterate for n_iter (N/b) times
            for i in range(n_iter):
                # Batch inputs
                start_idx = i*self.batch_sz
                end_idx = (i+1)*self.batch_sz
                if self.n_bits:
                    X = np.array(data_va['M_bin'][start_idx:end_idx], dtype=np.float32)
                else:
                    X = np.array(data_va['M'][start_idx:end_idx]).real.astype(np.float32)
                y = np.array(data_va['y'][start_idx:end_idx], dtype=np.float32)
                
                # Run model
                loss, yhat = sess.run(
                    [self.loss, self.logits],
                    feed_dict={
                        self.inputs: X,
                        self.targets: y
                    })
                
                # Compute SNR
                M = np.array(data_va['M'][start_idx:end_idx])
                
                S_hat = np.round(yhat)*M
                S_hat = S_hat.transpose(0,2,1)
                S_hat_i = np.array([istft(s) for s in S_hat])
                
                S_idx = i*self.batch_sz//data.n_noise
                S = data_va['S'][S_idx]
                S = S.T
                S_i = np.repeat(istft(S)[None, :], self.batch_sz, axis=0)
                
                sample_snr = compute_SNR(S_i, S_hat_i)
                
                # Record results
                signal_losses[i] = loss
                signal_snrs[i] = sample_snr
                
            return signal_losses.mean(), signal_snrs.mean()
        
        # Modify configuration for proper GPU memory usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        # Begin session
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            # Initialize saver and result arrays
            saver = tf.train.Saver(saver_dict(tf.trainable_variables()))  
            if self.is_restore:
                saver.restore(sess, '{}'.format(self.model_nm))
            
            self.tr_losses, self.va_losses, self.va_snrs = empty_array((3,self.n_epochs))
            
            # Iterate and run
            with tqdm(total=self.n_epochs, desc='Epoch') as pbar:
                for i in range(self.n_epochs):
                    self.tr_losses[i] = _train(data.train)
                    self.va_losses[i], self.va_snrs[i] = _validate(data.test)
                    tf.logging.debug('Epoch {} SNR: {:.2f}'.format(i, self.va_snrs[i]))
                    
                    pbar.update(1)
            
            self.model_nm =  mod_name(self.model_nm, self.n_epochs, self.is_pretrain, self.va_snrs.max(), self.learning_rate)
            # Save the model
            saver.save(sess, self.model_nm)
            tf.logging.info('Saving parameters to {}'.format(self.model_nm))
            
            # Save the data
            if not self.is_restore:
                with open('{}.pkl'.format(self.model_nm), 'wb') as f: 
                    pickle.dump(data, f)
                tf.logging.info('Saving data to {}.pkl'.format(self.model_nm))
            