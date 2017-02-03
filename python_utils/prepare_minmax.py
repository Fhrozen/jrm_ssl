import numpy as np
import h5py
import glob
import os
from .datastft import single_spectrogram, multi_spectrogram
from .getdata import getdata
from .createlists import create_lists
from sys import stdout

def prepare_minmax(pr, WindSamp, Shift):
    nflist, cflist = create_lists(pr.angle, pr.typf, 'trn', pr.cfdir, pr.nfdir)
    num_files = len(cflist)
    print 'Calculating Mean Values:'
    print '    Processing {} files:'.format(num_files)
    for idx in range (num_files):
        stdout.write('File:{}/{}\r'.format(idx, num_files ))
        stdout.flush()
        FreqSamp, outdata_seq, indata_seq  = getdata(nflist, cflist, idx) 
        indata, outdata, _ = datastft(indata_seq,outdata_seq,FreqSamp,WindSamp,Shift)
        """ Preparing Minimum Maximum """
        inmin_tmp = np.amin(indata, axis=0, keepdims = True)
        inmax_tmp = np.amax(indata, axis=0, keepdims = True)
        outmin_tmp = np.amin(outdata, axis=0, keepdims = True)
        outmax_tmp = np.amax(outdata, axis=0, keepdims = True)
                
        if idx == 0:
            indata_min = inmin_tmp
            indata_max = inmax_tmp       
            outdata_min = outmin_tmp
            outdata_max = outmax_tmp   
        else:
            indata_min = np.amin(np.concatenate((indata_min, inmin_tmp), axis=0), axis=0, keepdims = True)
            indata_max = np.amax(np.concatenate((indata_max, inmax_tmp), axis=0), axis=0, keepdims = True)     
            outdata_min = np.amin(np.concatenate((outdata_min, outmin_tmp), axis=0), axis=0, keepdims = True)
            outdata_max = np.amax(np.concatenate((outdata_max, outmax_tmp), axis=0), axis=0, keepdims = True)  
    print 'Minimum & Maximum processed, saving at {}'.format(pr.mymfile)
    in_minmax = np.concatenate((indata_min, indata_max), axis =0)
    out_mimax = np.concatenate((outdata_min, outdata_max), axis=0)
    if os.path.exists(pr.mymfile):
        os.remove(pr.mymfile)
    with h5py.File(pr.mymfile,'a') as f:
        dset = f.create_dataset("inminmax", data=in_minmax )
        dset = f.create_dataset("outminmax", data=out_mimax)
    return in_minmax, out_mimax
 
