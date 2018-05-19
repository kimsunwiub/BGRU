from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import numpy as np

class Lloyd_Max(object):
    def __init__(self, n_bits=4, perc=0.1, gmm=None):
        self.n_bits = n_bits
        self.perc = perc
        self.gmm = gmm
    
    def fit(self, X):
        # Initializations
        X = np.vstack(X).real
        n, m = X.shape
        random_indices = np.random.randint(n, size=int(n*self.perc))
        n_clusters = np.power(self.n_bits, 2)
        
        # Clustering
        X_subset = X[random_indices].reshape(-1,1)
        gmm = GaussianMixture(n_components=n_clusters).fit(X_subset)
        gmm.means_.sort(axis=0) # <Check> Does this work?
        self.gmm = gmm 
    
    def binarize(self, X):
        bit = np.ones((len(X)*self.n_bits,), dtype=np.int)
        for i,x in enumerate(X):
            n_bits_temp = self.n_bits
            while n_bits_temp > 0:
                temp = x % 2
                if temp == 0:
                    bit[i*self.n_bits+n_bits_temp-1] = -1
                x >>= 1
                n_bits_temp -= 1
        return bit
    
    def quantize(self, X, fit=False):
        if fit:
            self.fit(X)
            
        def _quantize(X):
            X_bin = []
            with tqdm(total=len(X), desc='quantizing') as pbar:
                for elem in X:
                    ni, mi = elem.shape
                    clusters = self.gmm.predict(elem.reshape(-1,1).real)
                    X_bin.append(self.binarize(clusters).reshape(ni, mi*self.n_bits))
                    pbar.update(1)
            return X_bin
            
        return _quantize(X)