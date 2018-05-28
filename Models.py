from GRU_Modifications import TanhGRUCell
from librosa.core import istft as istft
import tensorflow as tf
from tqdm import tqdm
import numpy as np

def empty_array(size):
    """
    Initialize an array of 'size' number of nans
    
    Params:
        size: Size of the array
    """
    return np.full(size, np.nan)
        
def saver_dict(trainable_variables):
    # <Refactor> Use tf scopes. Try not to make this a fn.
    names = [
        'W_gate_0', 'b_gate_0', 'W_cand_0', 'b_cand_0', 'W_gate_1', 
        'b_gate_1', 'W_cand_1', 'b_cand_1', 'W_out', 'b_out'
    ]
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

    # <Idea> Does Binary part also go here with pretraining?
    def __init__(self, perc, bptt, n_epochs, learning_rate, batch_sz, feat, n_bits, n_layers, state_sz, verbose, restore_model, dir_models):
        
        # <TODO>
        # Change '_more' to # epoch form
        """
        feat: Number of features / classes
        """
        # <TODO> Reorder param
        self.perc = perc
        self.bptt = bptt
        self.feat = feat
        self.n_bits = n_bits
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.batch_sz = batch_sz
        self.state_size = state_sz
        self.learning_rate = learning_rate
        self.dir_models = dir_models
        self.restore_model = restore_model
        
        tf.reset_default_graph()
        if verbose: tf.logging.set_verbosity(tf.logging.DEBUG)
        self.build_inputs()
        self.build_GRU()
        self.build_loss()
        self.build_optimizer()
    
    def build_inputs(self):
        self.inputs = tf.placeholder(tf.float32, 
                        [None, None, self.feat * self.n_bits]) 
        self.targets = tf.placeholder(tf.float32, 
                        [None, None, self.feat])
        
    def build_GRU(self):
        cell = tf.contrib.rnn.MultiRNNCell(
            [TanhGRUCell(self.state_size) for _ in range(self.n_layers)]
        )
            
        states_series, current_state = tf.nn.dynamic_rnn(cell, self.inputs, dtype=tf.float32)

        outputs = tf.reshape(states_series,[-1,self.state_size])

        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        W_out = tf.Variable(initializer([self.state_size, self.feat]))
        b_out = tf.Variable(initializer([self.feat]))
            
        self.logits = tf.sigmoid(tf.matmul(outputs, tf.tanh(W_out)) + tf.tanh(b_out))
        self.logits = tf.reshape(self.logits, [self.batch_sz, -1, self.feat])
        
    def build_loss(self):
        logits_1d = tf.reshape(self.logits, [-1])
        labels = tf.reshape(self.targets, [-1])
        self.loss = tf.reduce_mean(tf.square(tf.subtract(logits_1d, labels)))
        
    def build_optimizer(self):
        self.op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    
    def train(self, data):
        
        def mod_name(old_name):
            prev_n_epoch = int(old_name.split('_')[-1])
            name_format = '_'.join(old_name.split('_')[:-1])
            new_name = '{}_{}'.format(name_format, self.n_epochs + prev_n_epoch)
            return new_name
        
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
                X_bin = np.array(data_tr['M_bin'][start_idx:end_idx], dtype=np.float32)
                y = np.array(data_tr['y'][start_idx:end_idx], dtype=np.float32)
                
                # Run model w.r.t. lookback length
                if self.bptt == -1: local_bptt = X_bin[0].shape[0]
                else: local_bptt = self.bptt
                assert (local_bptt <= X_bin[0].shape[0])
                
                feed_sz = X_bin[0].shape[0]//local_bptt
                feed_losses = empty_array(feed_sz)
                
                for j in range(feed_sz):
                    start_idx = j*local_bptt
                    end_idx = (j+1)*local_bptt
                    X_feed = X_bin[:,start_idx:end_idx]
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
                X_bin = np.array(data_va['M_bin'][start_idx:end_idx], dtype=np.float32)
                y = np.array(data_va['y'][start_idx:end_idx], dtype=np.float32)
                
                # Run model
                loss, yhat = sess.run(
                    [self.loss, self.logits],
                    feed_dict={
                        self.inputs: X_bin,
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
            if self.restore_model:
                saver.restore(sess, '{}'.format(self.restore_model))
                save_nm = mod_name(self.restore_model)
            else:
                t_stamp = '{0:%m.%d(%H:%M)}'.format(datetime.now())
                save_nm = '{}/{}_lr({})_batchsz({})_bptt({})_data({})_noise({})_{}'.format(
                    self.dir_models, t_stamp, self.learning_rate, self.batch_sz, self.bptt, data.data_sz, data.n_noise, self.n_epochs)    
            
            self.tr_losses, self.va_losses, self.va_snrs = empty_array((3,self.n_epochs))
            
            # Iterate and run
            with tqdm(total=self.n_epochs, desc='Epoch') as pbar:
                for i in range(self.n_epochs):
                    self.tr_losses[i] = _train(data.train)
                    self.va_losses[i], self.va_snrs[i] = _validate(data.test)
                    tf.logging.debug('Epoch {} SNR: {:.2f}'.format(i, self.va_snrs[i]))
                    
                    pbar.update(1)
                    
            # Save the model
            saver.save(sess, save_nm)
            tf.logging.debug('Saving parameters to {}'.format(save_nm))
            