from matplotlib import gridspec, pyplot as plt
import tensorflow as tf
from argparse import ArgumentParser
from datetime import datetime
import _pickle as pickle
import logging
import os 
import re

from Models import GRU_Net
from DataSets import SourceSeparation_DataSet

def parse_arguments():
    parser = ArgumentParser()
    
    parser.add_argument("learning_rate", type=float, 
                        help="Learning rate")
    parser.add_argument("batch_sz", type=int, 
                        help="Batch size (multiples of number of noise signals)")
    parser.add_argument("bptt", type=int,
                        help="Back Propagation Through Time timestep")
    parser.add_argument("n_epochs", type=int, 
                        help="Number of epochs")
    parser.add_argument("gpu_id", type=int,
                        help="GPU ID")
    
    parser.add_argument("-z", "--beta1", type=float, default=0.9,
                        help="Beta1 value for AdamOptimizer (Default: 0.9)")
    parser.add_argument("-y", "--beta2", type=float, default=0.999,
                        help="Beta2 value for AdamOptimizer")
    
    parser.add_argument("-f", "--feat", type=int, default=513,
                        help="Number of features (Default: 513)")
    parser.add_argument("-l", "--n_layers", type=int, default=2,
                        help="Number of BGRU layers (Default: 2)")
    parser.add_argument("-e", "--is_pretrain", action='store_true',
                        help = "Use this option to pretrain")
    parser.add_argument("-o", "--model_nm", type=str, default=None,
                        help="Pretrained model to load")
    parser.add_argument("-k", "--pickled_data", type=str, default=None,
                        help="Pretrained data to load")
    parser.add_argument("-b", "--n_bits", type=int, default=None,
                        help="Number of bits used for quantization (Default: None)")
    parser.add_argument("-s", "--state_sz", type=int, default=1024,
                        help="Number of hidden units in each layer (Default: 1024)")
    parser.add_argument("-p", "--perc", type=float, default=0.1,
                        help="Proportion of data to sample for quantization (Default: 0.1)")
    parser.add_argument("-m", "--dir_models", type=str, default='Saved_Models',
                        help="Path to save Tensorflow model je m'appelle mercy (Default: 'Saved_Models')")
    parser.add_argument("-r", "--dir_results", type=str, default='Saved_Results',
                        help="Path to save GRU_Net model and plots from experiments (Default: 'Saved_Results')")
    parser.add_argument("-d", "--data_sz", type=str, default='small',
                        help="Size of data set (Default: small(2)). [Options: small(2), medium(4), large(8), xlarge(12)]")
    parser.add_argument("-n", "--n_noise", type=str, default='few',
                        help="Number of noise signals (Default: few(5)) [Options: few(5), many(10)]")
    
    parser.add_argument("-v", "--verbose",  action='store_true',
                        help = "Print SNR outputs from each epoch (Default: False)")

    return parser.parse_args()

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


def main():
    args = parse_arguments()
    logger = logging.getLogger(); logger.setLevel(logging.INFO)
    if args.pickled_data:
        with open(args.pickled_data, 'rb') as f:
            data = pickle.load(f)
        logger.info('Restoring data from {}'.format(args.pickled_data))
    else:
        data = SourceSeparation_DataSet(args.data_sz, args.n_noise)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    t_stamp = '{0:%m.%d(%H:%M)}'.format(datetime.now())
    
    is_restore = False
    if args.model_nm:
        is_restore = True
        run_info = mod_name(args.model_nm, args.n_epochs, args.is_pretrain, 1, args.learning_rate)
        args.model_nm = '{}/{}'.format(args.dir_models, args.model_nm)
    else:
        run_info = '{}_pretrain({})_batchsz({})_bptt({})_data({})_noise({})_bits({})'.format(
            t_stamp, args.is_pretrain, args.batch_sz, args.bptt, data.data_sz, data.n_noise, args.n_bits)
        args.model_nm = '{}/{}'.format(args.dir_models, run_info)
    
    model = GRU_Net(args.perc,                 
                    args.bptt,
                    args.n_epochs, 
                    args.learning_rate, 
                    args.batch_sz, 
                    args.feat,
                    args.n_layers,
                    args.state_sz,
                    args.verbose,
                    is_restore,
                    args.model_nm,
                    args.n_bits,
                    args.is_pretrain)
    model.train(data)

    if is_restore:
        args.n_epochs = 0
    if args.is_pretrain:
        plot_name = '{}/{}'.format(args.dir_results, mod_name(run_info, args.n_epochs, args.is_pretrain, model.va_snrs.max(), args.learning_rate))
    else:
        plot_name = '{}/{}'.format(args.dir_results, mod_name(run_info, args.n_epochs, args.is_pretrain, model.va_snrs.max()))
    plot_results(model, plot_name)
    
if __name__ == "__main__":
    main()