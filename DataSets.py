from librosa.core import stft as stft, load as load
from Quantizations import Lloyd_Max
from tqdm import tqdm
import numpy as np
import os

eps = 1e-7

class SourceSeparation_DataSet(object):
    def __init__(self, data_sz, n_noise):
        self.train = {}
        self.test = {}
        self.data_sz = data_sz
        self.n_noise = n_noise
        self.sr = 16000
        self.Lloyd_Max = Lloyd_Max()
        self.load()
        self.quantize()
    
    def load(self):
        def get_speakers():
            # <TODO> Dynamic Random Sampling
            return
        
        def get_dr1_speakers():
            path_train = 'Data/train/dr1' 
            path_test = 'Data/test/dr1' 
            
            speakers_train = ['fcjf0', 'mcpm0'] 
            speakers_test = ['faks0']
            if self.data_sz != 'small':
                speakers_train += ['fdaw0', 'mdac0']
                speakers_test += ['mdab0']
                if self.data_sz != 'medium':
                    speakers_train += ['fdml0', 'fecd0', 'mdpk0', 'medr0']
                    speakers_test += ['fdac1']
                    if self.data_sz != 'large':
                        speakers_train += ['fetb0', 'fjsp0', 'mgrl0', 'mjeb1']
                        speakers_test += ['felc0']
                        assert self.data_sz == 'xlarge'
            
            fullpath_train = ['%s/%s' % (path_train, s) for s in speakers_train]
            fullpath_test = ['%s/%s' % (path_test, s) for s in speakers_test]
            
            return fullpath_train, fullpath_test
        
        def get_noises():
            path_noise = 'Data/Duan'
            names = ['birds', 'casino', 'cicadas', 'computerkeyboard', 'eatingchips']     
            if self.n_noise != 'few':
                names += ['frogs', 'jungle', 'machineguns', 'motorcycles', 'ocean']
                assert self.n_noise == 'many'
            self.n_noise = len(names)
            return ['%s/%s.wav' % (path_noise, s) for s in names]

        def get_signals_from_speakers(speakers):    
            return [['%s/%s' % (s,q) for q in filter(lambda x: 'wav' in x, os.listdir(s))] for s in speakers]
        
        def get_data(signals):
            S_array, M_array, IBM_array = [], [], []

            def normalize_signal(sig):
                return (sig - sig.mean()) / sig.std()
            
            def create_ideal_binary_mask(S, N):
                M = S / (S + N + eps)
                M[np.where(M <= 0.5)] = 0
                M[np.where(M > 0.5)] = 1
                return M.real
            
            cnt = 1
            n_signals = len(signals)
            for signal in signals:
                signal_info = signal[0].split('/')
                status = signal_info[1]
                speaker = signal_info[3]
                
                with tqdm(total=len(signal), desc='%s(%d/%d)|%s' % (status, cnt, n_signals, speaker)) as pbar:
                    for fn in signal:
                        S = stft(normalize_signal(load(fn, sr=self.sr)[0]), n_fft=1024).T
                        S_array.append(S)

                        for fn in signals_noise:
                            N = stft(normalize_signal(load(fn, sr=self.sr)[0]), n_fft=1024).T
                            offset = np.random.randint(0,1000) # <Refactor>
                            N = N[offset:offset+S.shape[0]]

                            M_array.append(S+N)
                            IBM_array.append(create_ideal_binary_mask(S,N))
                            
                        pbar.update(1)
                cnt += 1
                
            if status == 'train':
                return {'M':M_array, 'y':IBM_array}
            return {'S':S_array, 'M':M_array, 'y':IBM_array}
        
        speakers_train, speakers_test = get_dr1_speakers()
        signals_noise = get_noises()
        
        signals_train = get_signals_from_speakers(speakers_train)
        signals_test = get_signals_from_speakers(speakers_test)
        
        self.train = get_data(signals_train)
        self.test = get_data(signals_test)
        
    def quantize(self):
        self.train['M_bin'] = self.Lloyd_Max.quantize(self.train['M'], fit=True)
        self.test['M_bin'] = self.Lloyd_Max.quantize(self.test['M'])