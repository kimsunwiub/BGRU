from librosa.core import istft as istft
from GRU_Modifications import TanhGRUCell
import tensorflow as tf
from tqdm import tqdm
import numpy as np

def empty_array(size):
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

def get_SNR(S, S_hat):
    """
    S: orignal source
    S_hat: recovered
    """
    return 10*np.log10(np.sum(np.power(S,2)) / np.sum(np.power(S - S_hat, 2)))
         
    
class GRU_Net(object):
    # <Idea> Does Binary part also go here with pretraining?
    def __init__(self, perc, bptt, n_epochs, learning_rate, batch_sz, feat, n_bits, n_layers, ste_sz, verbose, save_name):
        """
        feat: Number of features / classes
        """
        self.learning_rate = learning_rate
        self.save_name = save_name
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.state_size = ste_sz
        self.batch_sz = batch_sz
        self.n_bits = n_bits
        self.perc = perc
        self.bptt = bptt
        self.feat = feat
            
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
        
    def build_loss(self):
        logits_1d = tf.reshape(self.logits, [-1])
        labels = tf.reshape(self.targets, [-1])
        self.loss = tf.reduce_mean(tf.square(tf.subtract(logits_1d, labels)))
        
    def build_optimizer(self):
        self.op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    
    # <Idea> What if want to load a session?
    
    def train(self, data):
        
        def _train(data_tr):
            """
            v.1: No overlap, no batch, feed by bptt per signal.
            ->v.2: No overlap, no batch, feed by entire signal (bptt = n)
            """
            n_mixed = len(data_tr['M'])
            signal_losses, signal_snrs = empty_array((2,n_mixed))
                
            for tr_i in range(n_mixed):
                X_bin, y = data_tr['M_bin'][tr_i], data_tr['y'][tr_i]
                feed_sz = int(X_bin.shape[0]/self.bptt)
                feed_losses = empty_array(feed_sz)
                for j in range(feed_sz):
                    X_feed = X_bin[j*self.bptt:(j+1)*self.bptt]
                    y_feed = y[j*self.bptt:(j+1)*self.bptt]
                
                    _, loss = sess.run(
                        [self.op, self.loss],
                        feed_dict={
                            self.inputs: np.expand_dims(X_feed, 0),
                            self.targets: np.expand_dims(y_feed, 0)
                        })
                    feed_losses[j] = loss
                signal_losses[tr_i] = feed_losses.mean()
            return signal_losses.mean() # <TODO> Add batching
        
        def _validate(data_va):
            """
            v.1: Feed entire signal (bptt = n)
            -> v.2: Feed by bptt.
            """
            n_mixed = len(data_va['M'])
            signal_losses, signal_snrs = empty_array((2,n_mixed))
                
            for va_i in range(n_mixed):
                X_bin, y = data_va['M_bin'][va_i], data_va['y'][va_i]
                    
                loss, yhat = sess.run(
                    [self.loss, self.logits],
                    feed_dict={
                        self.inputs: np.expand_dims(X_bin, 0),
                        self.targets: np.expand_dims(y, 0)
                    })

                M = data_va['M'][va_i]
                S = data_va['S'][int(va_i/data.noise)]
                sample_snr = get_SNR(istft(S.T), istft((np.round(yhat)*M).T))

                signal_losses[va_i] = loss
                signal_snrs[va_i] = sample_snr
            return signal_losses.mean(), signal_snrs.mean()
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(saver_dict(tf.trainable_variables()))  
            self.tr_losses, self.va_losses, self.va_snrs = empty_array((3,self.n_epochs))

            with tqdm(total=self.n_epochs, desc='Epoch') as pbar:
                for epc_i in range(self.n_epochs):
                    self.tr_losses[epc_i] = _train(data.train)
                    self.va_losses[epc_i], self.va_snrs[epc_i] = _validate(data.test)
                    pbar.update(1)

                    tf.logging.debug('Epoch {} SNR: {:.2f}'.format(epc_i, self.va_snrs[epc_i]))
                    
            saver.save(sess, self.save_name) #<TODO> add more detail in name
    