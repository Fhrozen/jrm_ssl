import numpy as np
from scipy.signal import hamming
import numpy as np

def STFT(x, wlen, h, nfft, fs): 
    ########################################################
    #              Short-Time Fourier Transform            %
    #               with MATLAB Implementation             %
    #                  For Python                          %
    # Copier: Nelson Yalta                       11/03/15  %
    ########################################################
    # function: [stft, f, t] = stft(x, wlen, h, nfft, fs)
    # x - signal in the time domain
    # wlen - length of the hamming window
    # h - hop size
    # nfft - number of FFT points
    # fs - sampling frequency, Hz
    # f - frequency vector, Hz
    # t - time vector, s
    # stft - STFT matrix (only unique points, time across columns, freq across rows)
    # represent x as column-vector if it is not

    if (len(x.shape) > 1) and (x.shape[1] > 1):
            x = x.transpose()

    # length of the signal
    xlen = x.shape[0]

    # form a periodic hamming window
    win = hamming(wlen, False)
    # form the stft matrix
    rown = int(np.ceil((1.0+nfft)/2))
    coln = int(np.fix((xlen-wlen)/h) + 1)
    short_tft = np.zeros((rown,coln)).astype('complex64')

    # initialize the indexes
    indx = 0
    col = 0
    
    # perform STFT
    while (indx + wlen <= xlen):
        # windowing
        
        xw =x[indx:indx+wlen]*win
        
        # FFT
        X = np.fft.fft(xw,nfft)
     
        # update the stft matrix
        short_tft[:,col] = X[0:rown]

        # update the indexes
        indx +=  h
        col += 1
    
    # calculate the time and frequency vectors

    t = np.linspace(wlen/2,wlen/2+(coln-1)*h,coln)/fs
    f = np.arange(0,rown,dtype= np.float32)*fs/nfft   
    
    return short_tft, f, t
    
def single_spectrogram(inseq,fs,wlen,h,imag=False):
    """
        imag: Return Imaginary Data of the STFT on True 
    """
    NFFT = int(2**(np.ceil(np.log2(wlen)))) 
    K = np.sum(hamming(wlen, False))/wlen
    raw_data = inseq.astype('float32')
    raw_data = raw_data/np.amax(np.absolute(raw_data))
    stft_data,_,_ = STFT(raw_data,wlen,h,NFFT,fs)
    s = np.absolute(stft_data)/wlen/K;
    if np.fmod(NFFT,2):
        s[1:,:] *=2
    else:
        s[1:-2] *=2        
    real_data = np.transpose(20*np.log10(s + 10**-6)).astype(np.float32)
    if imag:
        imag_data = np.angle(stft_data).astype(np.float32)
        return real_data,imag_data 
    return real_data

def multi_spectrogram(inseq,fs,wlen,h):
    """
        inseq: audio data sequence in [# Channels, # Samples]
    """
    CHANNELS=inseq.shape[0]
    for ch in range(0,CHANNELS):
        spect_data = single_spectrogram(inseq[ch,:], fs, wlen, h)
        if ch==0:
            real_data = np.zeros(( spect_data.shape[0], CHANNELS,
                spect_data.shape[1]))
        real_data[:,ch,:] = spect_data
    return real_data


