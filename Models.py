from GRU_Modifications import TanhGRUCell, BinaryGRUCell, ScalingTanhGRUCell
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

    def __init__(self, bptt, n_epochs, learning_rate, beta1, beta2, batch_sz, verbose, is_restore, model_nm, n_bits, is_binary_phase, gain, clip_val, dropout1, dropout_cell, dropout2, sparsity_gru, sparsity_out, tensorboard, rs_rate):
        
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
        self.sparsity_gru = sparsity_gru
        self.sparsity_out = sparsity_out
        self.rs_rate = rs_rate
        
        self.dropout1 = dropout1
        self.dropout_cell = dropout_cell
        self.dropout2 = dropout2
        
        self.is_restore = is_restore
        self.is_binary_phase = is_binary_phase
        self.tensorboard = tensorboard
        
        if verbose: tf.logging.set_verbosity(tf.logging.DEBUG)
        else: tf.logging.set_verbosity(tf.logging.INFO)
        
        tf.reset_default_graph()
        self.build_inputs()
        self.build_GRU()
        self.build_loss()
        self.build_optimizer()
    
    def build_inputs(self):
        
        with tf.variable_scope("input_layer"):
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
            
            ## init_states = (tf.Variable(initializer([self.batch_sz, self.state_sz])), tf.Variable(initializer([self.batch_sz, self.state_sz])))
            
        else:    
            # Create weight and bias Variables for restoring
            with tf.variable_scope("out_layer"):
                W_out = tf.get_variable("W_out", [self.state_sz, self.feat])
                b_out = tf.get_variable("b_out", [self.feat])
            
            ## ini0 = tf.get_variable("ini_0", [self.batch_sz, self.state_sz])
            ## ini1 = tf.get_variable("ini_1", [self.batch_sz, self.state_sz])
            with tf.variable_scope("layer_1"):
                gk0 = tf.get_variable("gk", [self.feat*self.n_bits+self.state_sz, self.state_sz*2])
                gb0 = tf.get_variable("gb", [self.state_sz*2])
                ck0 = tf.get_variable("ck", [self.feat*self.n_bits+self.state_sz, self.state_sz])
                cb0 = tf.get_variable("cb", [self.state_sz])
            with tf.variable_scope("layer_2"):
                gk1 = tf.get_variable("gk", [self.state_sz*2, self.state_sz*2])
                gb1 = tf.get_variable("gb", [self.state_sz*2])
                ck1 = tf.get_variable("ck", [self.state_sz*2, self.state_sz])
                cb1 = tf.get_variable("cb", [self.state_sz])
            
            # <TODO> Command-line arguments for Wn, Wp
            with tf.variable_scope("layer_1_scaling"):
                p_g0 = tf.get_variable('p_g0', initializer=0.00975)
                n_g0 = tf.get_variable('n_g0', initializer=0.00990)
                p_g0_b = tf.get_variable('p_g0_b', initializer=1.0)
                n_g0_b = tf.get_variable('n_g0_b', initializer=1.0)
                p_c0 = tf.get_variable('p_c0', initializer=0.00870)
                n_c0 = tf.get_variable('n_c0', initializer=0.00820)
                p_c0_b = tf.get_variable('p_c0_b', initializer=0.00270)
                n_c0_b = tf.get_variable('n_c0_b', initializer=0.00275)
                ### p_ini0 = tf.get_variable('p_ini', initializer=0.???)
                ### n_ini0 = tf.get_variable('n_ini', initializer=0.???)
                
            with tf.variable_scope("layer_2_scaling"):
                p_g1 = tf.get_variable('p_g1', initializer=0.0470)
                n_g1 = tf.get_variable('n_g1', initializer=0.0478)
                p_g1_b = tf.get_variable('p_g1_b', initializer=1.0)
                n_g1_b = tf.get_variable('n_g1_b', initializer=1.0)
                p_c1 = tf.get_variable('p_c1', initializer=0.0470)
                n_c1 = tf.get_variable('n_c1', initializer=0.0475)
                p_c1_b = tf.get_variable('p_c1_b', initializer=0.0028)
                n_c1_b = tf.get_variable('n_c1_b', initializer=0.0029)
                ### p_ini1 = tf.get_variable('p_ini', initializer=0.???)
                ### n_ini1 = tf.get_variable('n_ini', initializer=0.???)
            
            with tf.variable_scope("layer_out_scaling"):
                p_out = tf.get_variable('p_out', initializer=0.045)
                n_out = tf.get_variable('n_out', initializer=0.045)
                p_out_b = tf.get_variable('p_out_b', initializer=0.049)
                n_out_b = tf.get_variable('n_out_b', initializer=0.0515)
             
            with tf.variable_scope("layer_1_sparsity"):
                sparse_gk0, q_err_gk0 = give_sparsity_two_th(gk0, self.sparsity_gru, p_g0, n_g0, self.rs_rate)
                sparse_gb0, q_err_gb0 = give_sparsity_two_th(gb0, self.sparsity_gru, p_g0_b, n_g0_b, self.rs_rate)
                sparse_ck0, q_err_ck0 = give_sparsity_two_th(ck0, self.sparsity_gru, p_c0, n_c0, self.rs_rate)
                sparse_cb0, q_err_cb0 = give_sparsity_two_th(cb0, self.sparsity_gru, p_c0_b, n_c0_b, self.rs_rate)

            with tf.variable_scope("layer_2_sparsity"):
                sparse_gk1, q_err_gk1 = give_sparsity_two_th(gk1, self.sparsity_gru, p_g1, n_g1, self.rs_rate)
                sparse_gb1, q_err_gb1 = give_sparsity_two_th(gb1, self.sparsity_gru, p_g1_b, n_g1_b, self.rs_rate)
                sparse_ck1, q_err_ck1 = give_sparsity_two_th(ck1, self.sparsity_gru, p_c1, n_c1, self.rs_rate)
                sparse_cb1, q_err_cb1 = give_sparsity_two_th(cb1, self.sparsity_gru, p_c1_b, n_c1_b, self.rs_rate)
            
            with tf.variable_scope("layer_out_sparsity"):
                sparse_W_out, q_err_W_out = give_sparsity_two_th(W_out, self.sparsity_out, p_out, n_out, self.rs_rate)
                sparse_b_out, q_err_b_out = give_sparsity_two_th(b_out, self.sparsity_out, p_out_b, n_out_b, self.rs_rate)
                
            cells = []
            cell = ScalingTanhGRUCell(self.state_sz, sparse_gk0, sparse_gb0, sparse_ck0, sparse_cb0)
            cells.append(cell)
            cell = ScalingTanhGRUCell(self.state_sz, sparse_gk1, sparse_gb1, sparse_ck1, sparse_cb1)
            cells.append(cell)
            multicell = tf.contrib.rnn.MultiRNNCell(cells)
            
            ## init_states = (ini0, ini1)
            ### init_states = (sparse_ini0, sparse_ini1)
        
        self.training = tf.placeholder(tf.bool)
        dropout_1 = tf.layers.dropout(self.inputs, self.dropout1, training=self.training)
        ## states_series, current_state = tf.nn.dynamic_rnn(multicell, dropout_1, initial_state=init_states, dtype=tf.float32)
        states_series, current_state = tf.nn.dynamic_rnn(multicell, dropout_1, dtype=tf.float32)
        dropout_2 = tf.layers.dropout(states_series, self.dropout2, training=self.training)
        outputs = tf.reshape(dropout_2, [-1,self.state_sz])

        if self.is_binary_phase:
            self.logits = tf.sigmoid(tf.matmul(outputs, tf.tanh(W_out)) + tf.tanh(b_out))
            self.logits = tf.reshape(self.logits, [self.batch_sz, -1, self.feat])
        else:
            logits = B_sigmoid(tf.matmul(outputs, sparse_W_out) + sparse_b_out)
            self.logits = tf.reshape(logits, [self.batch_sz, -1, self.feat])
        
            self.q_loss = tf.reduce_sum(q_err_gk0 + q_err_gb0 + q_err_ck0 + q_err_cb0 + 
                                        q_err_gk1 + q_err_gb1 + q_err_ck1 + q_err_cb1 + 
                                        q_err_W_out + q_err_b_out)
            # TB: q_loss
        
    def build_loss(self):

        logits_1d = tf.reshape(self.logits, [-1])
        labels = tf.reshape(self.targets, [-1])
        self.loss = tf.reduce_mean(tf.square(tf.subtract(logits_1d, labels)))
        # TB: loss
#         if not self.is_binary_phase:
#             self.loss += self.q_loss
        
    def build_optimizer(self):

        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2)
        self.gvs = opt.compute_gradients(self.loss)
        self.capped_gvs = [(tf.clip_by_value(grad, -self.clip_val, self.clip_val), var) for grad, var in self.gvs]
        self.op = opt.apply_gradients(self.capped_gvs)
    
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
                    
                    feed_dict_ = {self.inputs: X_feed,
                                self.targets: y_feed,
                                self.training: True
                                }
                    _, loss = sess.run(
                            [self.op, self.loss],
                            feed_dict=feed_dict_)
                    feed_losses[j] = loss                         
                    
                signal_losses[i] = feed_losses.mean()                    
            
            return signal_losses.mean()
        
        def _validate(data_va, epoch):
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
                feed_dict_ = {self.inputs: X,
                                self.targets: y,
                                self.training: False
                                }
                if self.tensorboard:
                    loss, yhat, summary = sess.run(
                        [self.loss, self.logits, merged],
                        feed_dict=feed_dict_)
                    
                    summary_writer.add_summary(summary, epoch * n_iter + i)
                    
                    if i == 0: # ow, resource exhausts
                        summary_2, summary_3 = sess.run(
                            [merged_2, merged_3],
                            feed_dict=feed_dict_)
                        summary_writer.add_summary(summary_2, epoch)
                        summary_writer.add_summary(summary_3, epoch)
                        
                else:
                    loss, yhat = sess.run(
                        [self.loss, self.logits],
                        feed_dict=feed_dict_)
                    
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
            
            if self.tensorboard: # Save summaries for Tensorboard
                # Save summaries of weights and scaling parameters
                names = ['W_out', 'b_out',
                         ##
                    'gk0','gb0', 'ck0', 'cb0',
                    'gk1','gb1', 'ck1', 'cb1']
                if not self.is_binary_phase:
                    names += ['p_g0', 'n_g0', 'p_g0_b', 'n_g0_b',
                              'p_c0', 'n_c0', 'p_c0_b', 'n_c0_b',
                              'p_g1', 'n_g1', 'p_g1_b', 'n_g1_b',
                              'p_c1', 'n_c1', 'p_c1_b', 'n_c1_b',
                              'p_out', 'n_out', 'p_out_b', 'n_out_b']
                
                for name,var in zip(names, tf.trainable_variables()):
                    with tf.variable_scope(name):
                        tf.summary.histogram(name, var)
                        tf.summary.scalar(name + '_mean', tf.reduce_mean(var))
                        tf.summary.scalar(name + '_min', tf.reduce_min(var))
                        tf.summary.scalar(name + '_max', tf.reduce_max(var))
                                    
                # Create FileWriters
                log_dir = "/N/u/kimsunw/workspace/BGRU/Saved_Logs"
                merged = tf.summary.merge_all()

                # Save gradients
                # TODO: Save grads for Wn,Wp
                with tf.variable_scope("grad"):
                    gk0_summary_op = tf.summary.histogram('gk0', self.gvs[2][0])
                    ck0_summary_op = tf.summary.histogram('ck0', self.gvs[4][0])
                    gk1_summary_op = tf.summary.histogram('gk1', self.gvs[6][0])
                    ck1_summary_op = tf.summary.histogram('ck1', self.gvs[8][0])
                    merged_2 = tf.summary.merge([gk0_summary_op, ck0_summary_op, gk1_summary_op, ck1_summary_op])

                with tf.variable_scope("clipgrad"):
                    gk0_summary_op = tf.summary.histogram('gk0', self.capped_gvs[2][0])
                    ck0_summary_op = tf.summary.histogram('ck0', self.capped_gvs[4][0])
                    gk1_summary_op = tf.summary.histogram('gk1', self.capped_gvs[6][0])
                    ck1_summary_op = tf.summary.histogram('ck1', self.capped_gvs[8][0])
                    merged_3 = tf.summary.merge([gk0_summary_op, ck0_summary_op, gk1_summary_op, ck1_summary_op])

                summary_writer = tf.summary.FileWriter(log_dir + '/test/name', sess.graph)
        
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
                    self.va_losses[i], self.va_snrs[i] = _validate(data.test, i)
                    # TB: SNR
                    tf.logging.debug('Epoch {} SNR: {:.3f} Err_tr: {:.3f} Err_va: {:.3f}'.format(i, self.va_snrs[i], self.tr_losses[i], self.va_losses[i]))
                    
                    pbar.update(1)
            
            # self.model_nm =  mod_name(self.model_nm, self.n_epochs, self.is_binary_phase, self.tr_snrs.max(), self.learning_rate,
            #               self.beta1, self.beta2, self.gain, self.clip_val, self.dropout1, self.dropout_cell, self.dropout2)
            # Save the model
            # self.model_nm = 'Saved_Models/lr{}_t_{}_betas{},{}_SNR{:.4f}'.format(self.learning_rate, self.sparsity_gru, self.beta1, self.beta2, self.va_snrs.max())
            # saver.save(sess, self.model_nm)
            # tf.logging.info('Saving parameters to {}'.format(self.model_nm))
            #print ("dropout {}_{}_{}: SNR: {}".format(self.dropout1, self.dropout_cell, self.dropout2, self.va_snrs.max()))
