#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, glob, h5py, time, shutil, argparse, ConfigParser
import numpy as np
import matplotlib.pyplot as plt
from sys import stdout
from python_utils.audio import *
from multiprocessing import Pool
from random import shuffle

datasize = 225*100
wlen = 512
shift = 160
frames = 20
CHANNELS = 16
skip = 30
PPM = 30
ROOM_DIMS = [5,5]
nfdir = '/export/db01/Local360d/data' 
cleandir = '/export/corpus'
IMP_PREFIX = '/export/db01/Local360d/data/TSP/Hearbo16ch/impulses/imp'
DBPREFIX = 'Hearbo16'
FILEPREFIX = '/export/db01/HEARBO16_MULTI'

current_dir = os.getcwd()
if not os.path.exists('temp'):
    os.makedirs('temp')

minmax = np.ones((2,257), dtype=np.float32)
minmax[0,:] *= -120.0
minmax[1,:] *= 120.0

def preparedata(_ANGLE):
    save_dir = '{}/{:03d}'.format(FILEPREFIX, _ANGLE)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    print('Prepare Data for Angle:{}'.format(_ANGLE))
    """ Process Data """
    actual_file = 0   
    actual_size = 0
    snr_idx = 0
    radius_idx = 0
    snr_vals = ['Clean'] + range(45,-35,-1)
    radius_vals = [1.5]

    list_file = '{}/trn_lists/d{:03d}.list'.format(nfdir,int(_ANGLE/10)*10)
    with open(list_file) as f:
        cflist = f.readlines()
    cflist = ['{}{}'.format(cleandir,i.replace('\r','').replace('\n','')) for i in cflist]
    shuffle(cflist)
    nflist = [['{}/temp/tmpin_D{:03d}_ch{:02d}.wav'.format(current_dir,_ANGLE,i)] for i in range (CHANNELS)]
    num_files = len(cflist) 

    while (datasize > actual_size):  
        if actual_file >= num_files: actual_file = 0
        if snr_idx >= len(snr_vals): snr_idx=0
        _output_tmp = '{}/temp/tmpin_D{:03d}'.format(current_dir, _ANGLE)
 
        type_noise = 'GaussianMulti'
        type_noise = None if type_noise =='GaussianSingle' else -1
        synth_sources(cflist[actual_file], IMP_PREFIX, CHANNELS, _ANGLE, type_noise, snr_vals[snr_idx], _output_tmp)

        os.system('./fft_generator.n {} 0 "POW" {}.wav > /dev/null 2>&1'.format(_ANGLE, _output_tmp))
        os.system('./fft_generator.n {} 1 "POW" {} > /dev/null 2>&1'.format(_ANGLE, cflist[actual_file]))
    
        with h5py.File('{}/temp/temp_D{:03d}_f00.h5'.format(current_dir, _ANGLE), 'r') as f: #INDATA
            indata = np.zeros(f['fft'].shape, dtype=np.float32)
            np.copyto(indata, f['fft'])
        with h5py.File('{}/temp/temp_D{:03d}_f01.h5'.format(current_dir, _ANGLE), 'r') as f: #OUTDATA
            outdata = np.zeros(f['fft'].shape, dtype=np.float32)
            np.copyto(outdata, f['fft'])

        if np.isinf(indata).any() or np.isinf(outdata).any():
            size = 0
            print("Excluding File: {}".format(nflist[0][actual_file]))
        else:  
            tmp_input, tmp_output, tmp_target= formatdata(frames, _ANGLE, indata,outdata= outdata,skip=skip)
            rms_tmp = np.sqrt(np.mean(np.square(tmp_output), axis=2))
            for lg in range (tmp_input.shape[0]):
                a = np.histogram(rms_tmp[lg], bins=[0,120,130])
                if a[0][1] > 13:
                    tmp_target[lg] = 360
            size = tmp_input.shape[0]
            if (actual_size+size) > datasize:
                nd = datasize - actual_size 
                tmp_input = tmp_input[0:nd]
                tmp_output = tmp_output[0:nd]
                tmp_target = tmp_target[0:nd]
            #"" Scaling ""
            inputdata, outputdata = scaling(minmax,tmp_input, outscale=minmax, outdata=tmp_output)   
            inputdata2 = inputdata[:,(0,2,4,6,8,10,12,14),:,:]
            targetdata = tmp_target.astype(np.int32)
            iters = inputdata.shape[0]

            for conns in range(iters):
                stdout.write('File: {}\r'.format(actual_size+conns))
                stdout.flush()
                fl_name = '{}/id{:06d}.h5'.format(save_dir,actual_size+conns)
                with h5py.File(fl_name, 'a') as f:
                    ds = f.create_dataset('arr_0', data = inputdata[conns])
                    ds = f.create_dataset('arr_1', data = outputdata[conns])
                    ds = f.create_dataset('arr_2', data = targetdata[conns])
                    ds = f.create_dataset('arr_3', data = inputdata2[conns])
        actual_size += size
        actual_file +=1   
        snr_idx +=1
    print('Angle:{:03d} Datasize:{}  files:{} '.format(_ANGLE,actual_size,actual_file))
    return       

if __name__ == '__main__':
    p = Pool(processes=5)
    p.map(preparedata,range(0,360,5))