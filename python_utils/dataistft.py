import numpy as np
from .istft import ISTFT
from scipy.signal import hamming


def dataistft(realdata,imgdata,fs,wlen,h):
    nfft = int(2**(np.ceil(np.log2(wlen))))

    K = np.sum(hamming(wlen, False))/wlen

    realdata = np.power(20,realdata/20) - 1e-6
    if np.fmod(nfft,2):
        realdata[1:-1,:] /=2 
    else:
        realdata[1:-2,:] /=2 
    realdata *= wlen*K
    prewav = realdata.transpose()*np.exp(1j*imgdata) 
    istft_data,_ = ISTFT(prewav, h, nfft, fs)
    max_dt = np.abs(istft_data).max()
    istft_data /= max_dt  
    return istft_data