import scipy.io.wavfile as wavfile
import numpy as np

def getdata(nflist, cflist, idx):    
    indata_seq = [] 
    for ch in range (len(nflist)):
        _, data = wavfile.read(nflist[ch][idx])
        if ch==0: 
        	indata_seq = np.zeros((len(nflist),data.shape[0]), dtype=np.float32)
        indata_seq[ch,:]= data.astype(np.float32)
    FreqSamp, outdata_seq = wavfile.read(cflist[idx]) 
    return FreqSamp, outdata_seq.astype(np.float32), indata_seq
