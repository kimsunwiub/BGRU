from DataSets import SourceSeparation_DataSet
from Models import GRU_Net
from utils import *

from argparse import ArgumentParser
from datetime import datetime
import _pickle as pickle
import os 

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
    
    parser.add_argument("-b", "--n_bits", type=int, default=4,
                        help="Number of bits used for quantization (Default: 4)")
    parser.add_argument("-c", "--clip_val", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("-d", "--tensorboard", action='store_true',
                        help="Save summaries for Tensorboard")
    parser.add_argument("-e", "--is_binary_phase", action='store_false',
                        help = "Use this option to go into binary traning phase")
    parser.add_argument("-g", "--gain", type=float, default=1.0,
                        help="GAIN to influence initialization stddev for the first layer of weights")
    parser.add_argument("-j", "--dropout1", type=float, default=0.0,
                        help="Dropout value for inputs (Default: 0.0)")
    parser.add_argument("-k", "--dropout_cell", type=float, default=0.0,
                        help="Dropout value for GRU Cells (Default: 0.0)")
    parser.add_argument("-l", "--dropout2", type=float, default=0.0,
                        help="Dropout value for outputs (Default: 0.0)")
    parser.add_argument("-o", "--model_nm", type=str, default=None,
                        help="Pretrained model to load")
    parser.add_argument("-r", "--rs_rate", type=float, default=0.10,
                        help="Random sampling rate for sparsity (Default: 0.10)")
    parser.add_argument("-t", "--scale_t", type=float, default=0.95,
                        help="Sparsity threshold (Default: 0.95)")
    parser.add_argument("-v", "--verbose",  action='store_true',
                        help = "Print SNR outputs from each epoch (Default: False)")
    parser.add_argument("-y", "--beta1", type=float, default=0.9,
                        help="Beta1 value for AdamOptimizer (Default: 0.9)")
    parser.add_argument("-z", "--beta2", type=float, default=0.999,
                        help="Beta2 value for AdamOptimizer (Default: 0.999)")

    return parser.parse_args()


def main():
    args = parse_arguments()
    with open('data_l_f.pkl', 'rb') as f: data = pickle.load(f) 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    t_stamp = '{0:%m.%d(%H:%M)}'.format(datetime.now())
    dir_models = 'Saved_Models'; dir_results = 'Saved_Results'
    
    is_restore = False
    if args.model_nm:
        is_restore = True
        run_info = mod_name(args.model_nm, args.n_epochs, args.is_binary_phase, 1, args.learning_rate,
                            args.beta1, args.beta2, args.gain)
        args.model_nm = '{}/{}'.format(dir_models, args.model_nm)
    else:
        run_info = '{}_pretrain({})_batchsz({})_bptt({})_bits({})'.format(
            t_stamp, args.is_binary_phase, args.batch_sz, args.bptt, args.n_bits)
        args.model_nm = '{}/{}'.format(dir_models, run_info)

    model = GRU_Net(args.bptt,
                    args.n_epochs, 
                    args.learning_rate, 
                    args.beta1, 
                    args.beta2,
                    args.batch_sz,
                    args.verbose,
                    is_restore,
                    args.model_nm,
                    args.n_bits,
                    args.is_binary_phase,
                    args.gain,
                    args.clip_val,
                    args.dropout1,
                    args.dropout_cell,
                    args.dropout2,
                    args.scale_t,
                    args.tensorboard,
                    args.rs_rate)
    model.train(data)

    if is_restore:
        args.n_epochs = 0
    #plot_name = '{}/{}'.format(dir_results, mod_name(run_info, args.n_epochs, args.is_binary_phase, model.va_snrs.max(), args.learning_rate, args.beta1, args.beta2, args.gain, args.clip_val, args.dropout1, args.dropout_cell, args.dropout2))
    plot_name = 'Saved_Results/lr{}_t_{}_betas{},{}_clip{}_SNR{:.4f}'.format(args.learning_rate, args.scale_t, args.beta1, args.beta2, args.clip_val, model.va_snrs.max())
    plot_results(model, plot_name)
    
if __name__ == "__main__":
    main()
