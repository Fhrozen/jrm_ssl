########################################################
#              Short-Time Fourier Transform            %
#               with MATLAB Implementation             %
#                  For Python                          %
# Copier: Nelson Yalta                       11/03/15  %
########################################################
from scipy.signal import hamming
import numpy as np

def ISTFT(data, h, nfft, fs):
    # function: [x, t] = istft(stft, h, nfft, fs)
    # stft - STFT matrix (only unique points, time across columns, freq across rows)
    # h - hop size
    # nfft - number of FFT points
    # fs - sampling frequency, Hz
    # x - signal in the time domain
    # t - time vector, s

    # estimate the length of the signal
    coln = data.shape[1]
    xlen = nfft + (coln-1)*h
    x = np.zeros((xlen,))

    # form a periodic hamming window
    win = hamming(nfft, False)

    # perform IFFT and weighted-OLA
    if np.fmod(nfft,2):
        lst_idx = -1
    else:
        lst_idx = -2

    for b in range (0, h*(coln-1),h):
        # extract FFT points
        X = data[:,1+b/h] 
        X = np.concatenate((X, np.conjugate(X[lst_idx:0:-1])))

        # IFFT
        xprim = np.real(np.fft.ifft(X))

        # weighted-OLA
        x[b:b+nfft] = x[b:b+nfft] + np.transpose(xprim*win)
    
    W0 = np.sum(win*win)
    x *= h/W0
    # calculate the time vector
    actxlen = x.shape[0]
    t = np.arange(0,actxlen-1,dtype=np.float32)/fs
    return x, t
