from matplotlib import gridspec, pyplot as plt
from datetime import datetime
import logging
import _pickle as pickle
import tensorflow as tf
from tensorflow.python.ops import math_ops
import time
import os 
import re

from DataSets import SourceSeparation_DataSet

# DataGen

def make_and_save_data(data_sz, n_noise, perc):
    data = SourceSeparation_DataSet(data_sz, n_noise, perc=perc, is_quantize=True)
    start_time = time.time()
    d_dic = {'small': 's', 'medium': 'm', 'large': 'l', 'xlarge': 'xl'}
    n_dic = {'few': 'f', 'many': 'm'}
    file_nm = 'data_{}_{}.pkl'.format(d_dic[data_sz], n_dic[n_noise])
    with open (file_nm, 'wb') as f:
        pickle.dump(data, f)

start_time = time.time()
make_and_save_data('xlarge', 'many', 1)
end_time = time.time()

print (end_time - start_time)
