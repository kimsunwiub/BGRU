from GRU_Modifications import TanhGRUCell, BinaryGRUCell
from Initializers import my_xavier_initializer
from tensorflow.python.ops import math_ops
from librosa.core import istft as istft
import _pickle as pickle
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import re

from utils import *
    
class GRU_Net(object):

    def __init__(self, bptt, n_epochs, learning_rate, beta1, beta2, batch_sz, verbose, is_restore, model_nm, n_bits, is_binary_phase, gain, clip_val, dropout1, dropout_cell, dropout2, rho):
        
        self.feat = 513
        self.n_layers = 2
        self.state_sz = 1024
        
        self.bptt = bptt
        self.gain = gain
        self.beta1 = beta1
        self.beta2 = beta2
        self.n_bits = n_bits
        self.clip_val = clip_val
        self.model_nm = model_nm
        self.n_epochs = n_epochs
        self.batch_sz = batch_sz
        self.learning_rate = learning_rate
        self.rho = rho
        
        self.dropout1 = dropout1
        self.dropout_cell = dropout_cell
        self.dropout2 = dropout2
        
        self.is_restore = is_restore
        self.is_binary_phase = is_binary_phase
        
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
        
        if self.is_binary_phase:
            cells = []
            my_weight_initializer=my_xavier_initializer(uniform=False, gain=self.gain)
            weight_initializer=tf.contrib.layers.xavier_initializer(uniform=False)

            cell = TanhGRUCell(self.state_sz, kernel_initializer=my_weight_initializer)
            dropout=self.dropout_cell
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
            cells.append(cell)
            cell = TanhGRUCell(self.state_sz, kernel_initializer=weight_initializer)
            cells.append(cell)       
            multicell = tf.contrib.rnn.MultiRNNCell(cells)

            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            W_out = tf.Variable(initializer([self.state_sz, self.feat]))
            b_out = tf.Variable(initializer([self.feat]))
            #init_state = (tf.Variable(initializer([self.batch_sz, self.state_sz])), tf.Variable(initializer([self.batch_sz, self.state_sz])))
            
        else:    
            # Create weight and bias Variables for restoring
            W_out = tf.get_variable("W_out", [self.state_sz, self.feat])
            b_out = tf.get_variable("b_out", [self.feat])
            
            # ini_0 = tf.get_variable("ini_0", [self.batch_sz, self.state_sz])
            # ini_1 = tf.get_variable("ini_1", [self.batch_sz, self.state_sz])
            
            gk0 = tf.get_variable("gk0", [self.feat*self.n_bits+self.state_sz, self.state_sz*2])
            gb0 = tf.get_variable("gb0", [self.state_sz*2])
            ck0 = tf.get_variable("ck0", [self.feat*self.n_bits+self.state_sz, self.state_sz])
            cb0 = tf.get_variable("cb0", [self.state_sz])

            gk1 = tf.get_variable("gk1", [self.state_sz*2, self.state_sz*2])
            gb1 = tf.get_variable("gb1", [self.state_sz*2])
            ck1 = tf.get_variable("ck1", [self.state_sz*2, self.state_sz])
            cb1 = tf.get_variable("cb1", [self.state_sz])
            
            # Initial Tanh operation
            tanh_W_out = tf.tanh(W_out)
            tanh_b_out = tf.tanh(b_out)
            
            tanh_gk0 = tf.tanh(gk0)
            tanh_gb0 = tf.tanh(gb0)
            tanh_ck0 = tf.tanh(ck0)
            tanh_cb0 = tf.tanh(cb0)

            tanh_gk1 = tf.tanh(gk1)
            tanh_gb1 = tf.tanh(gb1)
            tanh_ck1 = tf.tanh(ck1)
            tanh_cb1 = tf.tanh(cb1)
            
            cells = []
            cell = BinaryGRUCell(self.state_sz, tanh_gk0, tanh_gb0, tanh_ck0, tanh_cb0, self.rho)
            cells.append(cell)
            cell = BinaryGRUCell(self.state_sz, tanh_gk1, tanh_gb1, tanh_ck1, tanh_cb1, self.rho)
            cells.append(cell)
            multicell = tf.contrib.rnn.MultiRNNCell(cells)
            
            # init_state = (ini_0, ini_1)
        
        #states_series, current_state = tf.nn.dynamic_rnn(multicell, self.inputs, initial_state=init_state, dtype=tf.float32)
        self.training = tf.placeholder(tf.bool)
        dropout_1 = tf.layers.dropout(self.inputs, self.dropout1, training=self.training)
        states_series, current_state = tf.nn.dynamic_rnn(multicell, dropout_1, dtype=tf.float32)
        dropout_2 = tf.layers.dropout(states_series, self.dropout2, training=self.training)
        outputs = tf.reshape(dropout_2, [-1,self.state_sz])

        if self.is_binary_phase:
            self.logits = tf.sigmoid(tf.matmul(outputs, tf.tanh(W_out)) + tf.tanh(b_out))
            self.logits = tf.reshape(self.logits, [self.batch_sz, -1, self.feat])
        else:
            mask_w = get_mask(tanh_W_out, self.rho)
            mask_b = get_mask(tanh_b_out, self.rho)
            w_ = tf.where(mask_w, B_tanh(tanh_W_out), tf.zeros(tanh_W_out.shape))
            b_ = tf.where(mask_b, B_tanh(tanh_b_out), tf.zeros(tanh_b_out.shape))
            logits = B_sigmoid(tf.matmul(outputs, w_) + b_)
            self.logits = tf.reshape(logits, [self.batch_sz, -1, self.feat])
        
    def build_loss(self):

        logits_1d = tf.reshape(self.logits, [-1])
        labels = tf.reshape(self.targets, [-1])
        self.loss = tf.reduce_mean(tf.square(tf.subtract(logits_1d, labels)))
        
    def build_optimizer(self):

        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2)
        gvs = opt.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -self.clip_val, self.clip_val), var) for grad, var in gvs]
        self.op = opt.apply_gradients(capped_gvs)
    
    def train(self, data):
        
        def _train(data_tr):
            """
            Helper function to train(): Training
            """
            # Initialize result arrays
            n_iter = len(data_tr['M'])//self.batch_sz
            #print (n_iter)
            
            # -- 5 for short DEBUG -->
            #n_iter = 1
            # < --
            
            signal_losses = empty_array(n_iter)
#             signal_snrs = empty_array(n_iter)

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
                local_n = X[0].shape[0]
                if self.bptt == -1: 
                    local_bptt = local_n
                    if local_bptt > 200: # max for large many
                        local_bptt = 200
                else: local_bptt = self.bptt
                assert (local_bptt <= local_n)
                
                feed_sz = local_n//local_bptt
                feed_losses = empty_array(feed_sz)
                
                for j in range(feed_sz):
                    start_idx_j = j*local_bptt
                    end_idx_j = (j+1)*local_bptt
                    X_feed = X[:,start_idx_j:end_idx_j]
                    y_feed = y[:,start_idx_j:end_idx_j]
                    
                    _, loss = sess.run(
                        [self.op, self.loss],
                        feed_dict={
                            self.inputs: X_feed,
                            self.targets: y_feed,
                            self.training: True
                        })
                    feed_losses[j] = loss
                    
                    
#                 # -- DEBUG -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- 
#                 # -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- 
                    
#                 # Compute SNR on training phase to see faster if it works
#                 # Run model
#                 yhat = sess.run(
#                     [self.logits],
#                     feed_dict={
#                         self.inputs: X,
#                         self.targets: y,
#                         self.training: False
#                     })[0]
              
#                 # Compute SNR
#                 M = np.array(data_tr['M'][start_idx:end_idx])
                
#                 S_hat = np.round(yhat)*M
#                 S_hat = S_hat.transpose(0,2,1)
#                 S_hat_i = np.array([istft(s) for s in S_hat])
                
#                 S_idx = i*self.batch_sz//data.n_noise
#                 S = data_tr['S'][S_idx]
#                 S = S.T
#                 S_i = np.repeat(istft(S)[None, :], self.batch_sz, axis=0)
                
#                 sample_snr = compute_SNR(S_i, S_hat_i)
#                 print ("LR {} Clip {} Rho {}: SNR: {:.4f}".format(
#                     self.learning_rate, self.clip_val, self.rho, sample_snr))
#                 # < -- DEBUG
                

#                 signal_snrs[i] = sample_snr
#             return signal_losses.mean(), signal_snrs.mean()
#             print (signal_snrs.mean())
                signal_losses[i] = feed_losses.mean()
            return signal_losses.mean()
        
        def _validate(data_va):
            """
            Helper function to train(): Validation
            """
            # Initialize result arrays
            n_iter = len(data_va['M'])//self.batch_sz
            #n_iter = 1
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
                        self.targets: y,
                        self.training: False
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
            self.tr_snrs = empty_array(self.n_epochs)
            with tqdm(total=self.n_epochs, desc='Epoch') as pbar:
                for i in range(self.n_epochs):
                    self.tr_losses[i] = _train(data.train)
                    self.va_losses[i], self.va_snrs[i] = _validate(data.test)
                    tf.logging.debug('Epoch {} SNR: {:.3f} Err_tr: {:.3f} Err_va: {:.3f}'.format(i, self.va_snrs[i], self.tr_losses[i], self.va_losses[i]))
                    
                    pbar.update(1)
            
            # self.model_nm =  mod_name(self.model_nm, self.n_epochs, self.is_binary_phase, self.va_snrs.max(), self.learning_rate,
            #self.model_nm =  mod_name(self.model_nm, self.n_epochs, self.is_binary_phase, self.tr_snrs.max(), self.learning_rate,
            #                self.beta1, self.beta2, self.gain, self.clip_val, self.dropout1, self.dropout_cell, self.dropout2)
            # Save the model
            self.model_nm = 'Saved_Models/lr{}_rho_{}_SNR{:.4f}'.format(self.learning_rate, self.rho, self.va_snrs.max())
            saver.save(sess, self.model_nm)
            tf.logging.info('Saving parameters to {}'.format(self.model_nm))