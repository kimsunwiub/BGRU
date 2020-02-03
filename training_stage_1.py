import os
import numpy as np
import pickle
import time
from argparse import ArgumentParser

import tensorflow as tf

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

tf.compat.v1.disable_eager_execution()
@tf_export("nn.rnn_cell.GRUCell")
class JustGRUCell(LayerRNNCell):
  def __init__(self,
               num_units,
               gk0,
               gb0,
               ck0,
               cb0,
               activation=None,
               reuse=None):
    super(JustGRUCell, self).__init__(_reuse=reuse, name=None)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = None
    self._bias_initializer = None
    self.gk0 = gk0
    self.gb0 = gb0
    self.ck0 = ck0
    self.cb0 = cb0

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
    
    gate_inputs = tf.matmul(array_ops.concat([inputs, state], 1), self.gk0)
    gate_inputs = nn_ops.bias_add(gate_inputs, self.gb0)

    value = tf.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state
    
    candidate = tf.matmul(array_ops.concat([inputs, r_state], 1), self.ck0)
    candidate = nn_ops.bias_add(candidate, self.cb0)

    c = tf.tanh(candidate)
    new_h = (1 - u) * state + u * c
    return new_h, new_h

def parse_arguments():
    parser = ArgumentParser()
    
    parser.add_argument("node_id", type=int,
                        help="Node ID")
    parser.add_argument("gpu_id", type=int,
                        help="GPU ID")
    parser.add_argument("num_epochs", type=int,
                        help="Number of epochs")
    parser.add_argument("learning_rate", type=float, 
                        help="Learning rate")
    parser.add_argument("beta1", type=float,
                        help="Beta1 value for AdamOptimizer")
    parser.add_argument("beta2", type=float,
                        help="Beta2 value for AdamOptimizer")
    parser.add_argument("bptt", type=int,
                        help="BPTT Value")
    parser.add_argument("save_model", type=int,
                        help="Decide Saving Weights. (1: True, Else False)")
    
    parser.add_argument("-j", "--dropout1", type=float, default=0.05,
                        help="Dropout value for inputs (Default: 0.05)")
    parser.add_argument("-l", "--dropout2", type=float, default=0.20,
                        help="Dropout value for outputs (Default: 0.20)")
    parser.add_argument("-m", "--dropout_cell", type=float, default=0.20,
                        help="Dropout value for GRU Cell (Default: 0.20)")
    parser.add_argument("-s", "--bs", type=int, default=10,
                        help="Batch Size (Default: 10)")
    
    return parser.parse_args()

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

save_nm = "Phase_1_Ep{}".format(args.num_epochs)

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

gk0_tanh, gb0_tanh, ck0_tanh, cb0_tanh = tf.tanh(gk0), tf.tanh(gb0), tf.tanh(ck0), tf.tanh(cb0)

GRUcell = JustGRUCell(H, gk0_tanh, gb0_tanh, ck0_tanh, cb0_tanh)

# GRU Cell
dropout_1 = tf.compat.v1.layers.dropout(inputs, args.dropout1)
states_series, current_state = tf.compat.v1.nn.dynamic_rnn(GRUcell, dropout_1, dtype=tf.float32)
dropout_2 = tf.compat.v1.layers.dropout(states_series, args.dropout2)
outputs = tf.reshape(dropout_2, [-1, H])

# Linear output
W_out_tanh, b_out_tanh = tf.tanh(W_out), tf.tanh(b_out)
logits = tf.tanh(tf.matmul(outputs, W_out_tanh) + b_out_tanh)
trYh = tf.reshape(logits, [bs, -1, D_out])

# # Training loss
tr_err = tf.reduce_sum(tf.square(tf.subtract(trYh, target)))

# Optimizer
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2)
gvs = opt.compute_gradients(tr_err)
op = opt.apply_gradients(gvs)

tot_tr_err,tot_te_err,tot_te_snr = [],[],[]
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

with tf.compat.v1.Session() as sess:
    tic = time.time()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver(saver_dict_p1(tf.compat.v1.trainable_variables()))
    with tqdm(total=args.num_epochs, desc='GPU({}) Epoch'.format(args.gpu_id)) as pbar:
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
            pbar.update(1)
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
            pbar.update(1)
            print ("Epoch {} Te_Err {} Te_SDR {}".format(e, mean_te_err, mean_te_snr))
        
    print('Saving as {}. Avg SNR: {:.4f}'.format(save_nm, np.array(tot_te_snr).mean()))
    print ("args.save_model: {}".format(args.save_model))
    saver.save(sess, 'Saved_Models/{}'.format(save_nm))