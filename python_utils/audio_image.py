import scipy.io.wavfile as wavfile
from scipy.signal import hamming
import numpy as np
import matplotlib.pyplot as plt
from stft import STFT

def wav_to_image(filename, wlen, mindata, maxdata, save=False, name_save=None, ):
	h = wlen/4
	K = np.sum(hamming(wlen, False))/wlen

	nfft = int(2**(np.ceil(np.log2(wlen))))
	Fs, data_seq = wavfile.read(filename) 
	raw_data = data_seq.astype('float32')
	max_dt = np.amax(np.absolute(raw_data))
	raw_data = raw_data/max_dt
	stft_data,_,_ = STFT(raw_data,wlen,h,nfft,Fs)
	s = abs(stft_data)/wlen/K;
	if np.fmod(nfft,2):
	    s[1:,:] *=2
	else:
	    s[1:-2] *=2        
	data_temp = 20*np.log10(s + 10**-6)
	outdata = data_temp.transpose()

	"""Scaling"""
	mindata = np.amin(outdata, axis=0, keepdims = True)
	maxdata = np.amax(outdata, axis=0, keepdims = True)
	outdata -=mindata
	outdata /=(maxdata-mindata)
	outdata *=0.8
	outdata +=0.1
	figmin = np.zeros((5,outdata.shape[1]))
	figmax = np.ones((5,outdata.shape[1]))
	outdata = np.concatenate((outdata,figmin,figmax), axis=0)

	dpi = 96
	a = float(outdata.shape[0])/dpi
	b = float(outdata.shape[1])/dpi

	f = plt.figure(figsize=(b,a), dpi=dpi)
	f.figimage(outdata)
	if save:
		f.savefig(name_save, dpi=f.dpi)
	return f