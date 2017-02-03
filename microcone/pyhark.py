import harkpython.harkbasenode as harkbasenode
import numpy as np
import h5py, os
from scipy.signal import hamming

class StackFFT(harkbasenode.HarkBaseNode):
    def __init__(self):
        self.outputNames=("OUTPUT",) # one output terminal named "output"
        self.outputTypes=("prim_int",) # the type is primitive int.
        self.data_array = []
        self.prefix = './temp'
        self.wlen = 512
        self.K = np.sum(hamming(self.wlen, False))/self.wlen

    def calculate(self):
        data = np.asarray(np.absolute(self.INPUT),dtype=np.float32)
        data = data/self.wlen/self.K;
        data[:, 1:-2] *=2  
        data = 20*np.log10(data + 10**-6)
        if self.count == 0:
            self.data_array = data[None, :,:]
        else:
            self.data_array = np.concatenate((self.data_array, data[None, :,:])) 
        self.outputValues["OUTPUT"] = 1

    def __del__(self):
        filename = '{}/temp_D{:03d}_f{:02d}.h5'.format(self.prefix, self.EXT1, self.EXT2)
        if os.path.exists(filename):
           os.remove(filename)
        with h5py.File(filename, 'a') as f:
            d = f.create_dataset('fft', data=self.data_array)
        pass
