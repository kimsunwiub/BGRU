from librosa.core import stft as stft, load as load
from Quantizations import Lloyd_Max
from tqdm import tqdm
import numpy as np
import os

EPS = 1e-7

class SourceSeparation_DataSet(object):
    def __init__(self, data_sz, n_noise, n_bits=4, perc=0.1, is_quantize=False):
        self.train = {}
        self.test = {}
        self.data_sz = data_sz
        self.n_noise = n_noise
        self.sr = 16000
        self.Lloyd_Max = Lloyd_Max(n_bits, perc)
        self.load()
        if is_quantize:
            self.quantize()
    
    def load(self):
        def get_speakers():
            # <TODO> Random Sampling
            # <TODO> Directory Input
            return
        
        def get_Profs_speakers():
            # Speakers used for Professor's Notebook "Bitwise GRU - 1R"
            fullpath_train = ['Data/train/dr2/fcaj0', 'Data/train/dr1/mrcg0', 'Data/train/dr2/mjmd0', 
                              'Data/train/dr3/fmjf0', 'Data/train/dr4/fpaf0', 'Data/train/dr7/mclk0', 
                              'Data/train/dr7/mjai0', 'Data/train/dr2/fsrh0', 'Data/train/dr2/feac0', 
                              'Data/train/dr6/mcae0', 'Data/train/dr3/mreh1', 'Data/train/dr2/mrjt0'] 
            fullpath_test = ['Data/test/dr2/mcem0', 'Data/test/dr4/mljb0', 'Data/test/dr5/mrpp0', 'Data/test/dr7/fdhc0']
            return fullpath_train, fullpath_test
        
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
                M = S / (S + N + EPS)
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
        
        speakers_train, speakers_test = get_Profs_speakers() # <TODO> So only XL and M option technically for now
        signals_noise = get_noises()
        
        signals_train = get_signals_from_speakers(speakers_train)
        signals_test = get_signals_from_speakers(speakers_test)
        
        self.train = get_data(signals_train)
        self.test = get_data(signals_test)
        
    def quantize(self):
        self.train['M_bin'] = self.Lloyd_Max.quantize(self.train['M'], fit=True)
        self.test['M_bin'] = self.Lloyd_Max.quantize(self.test['M'])