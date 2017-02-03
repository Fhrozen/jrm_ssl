#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, h5py, glob, imp, shutil
import timeit, chainer, ConfigParser, itertools
from time import localtime, strftime, sleep
import numpy as np
from chainer import cuda, serializers
from multiprocessing import Pool, freeze_support
from sys import stdout
import argparse
import progressbar as prg
from progressbar import SimpleProgress
from python_utils.audio import *

parser = argparse.ArgumentParser(description='Evaluate trained model for Sound Source Localization')
parser.add_argument('exp_date', metavar='E', type=str, help='String for the experiment day')
args = parser.parse_args()
config = ConfigParser.ConfigParser()
typeNoise = 'SameNoise'
idxNoise = None
minmax = np.ones((2,257), dtype=np.float32)
minmax[0,:] *= -120.0
minmax[1,:] *= 120.0
_MINMAX = minmax

def prepared_data(args):
    _ANGLE, _CLEANFILE, _SNR = args
    _output_tmp = 'temp/tmpin_D{:03d}'.format(_ANGLE)
    synth_sources(_CLEANFILE, IMP_PREFIX, CHANNELS_SND, _ANGLE, idxNoise, _SNR, _output_tmp)
    _prepared = np.array((5,), dtype=np.object)
    os.system('./fft_generator.n {} 0 "POW" {}.wav > /dev/null 2>&1'.format(_ANGLE, _output_tmp)) 
    os.system('./fft_generator.n {} 1 "POW" {} > /dev/null 2>&1'.format(_ANGLE, _CLEANFILE)) 
    with h5py.File('temp/temp_D{:03d}_f00.h5'.format(_ANGLE), 'r') as f: #INDATA
        indata = np.zeros(f['fft'].shape, dtype=np.float32)
        np.copyto(indata, f['fft'])
    with h5py.File('temp/temp_D{:03d}_f01.h5'.format(_ANGLE), 'r') as f: #OUTDATA
        outdata = np.zeros(f['fft'].shape, dtype=np.float32)
        np.copyto(outdata, f['fft'])    
    indata, tmp_output, target= formatdata(frames, _ANGLE, indata,outdata= outdata,skip=skip, test_p=True) 
    rms_tmp = np.sqrt(np.mean(np.square(tmp_output), axis=2))
    for lg in range (tmp_output.shape[0]):
        a = np.histogram(rms_tmp[lg], bins=[0,120,130])
        if a[0][1] > 13:
            target[lg] = 360 
    if _DIFF:
        indata=scaling(_MINMAX,indata[:,_MICS],do_out=False)
    else:
        indata=scaling(_MINMAX,indata,do_out=False)
    if os.path.exists('temp/data_D{:03d}.h5'.format(_ANGLE)):
        os.remove('temp/data_D{:03d}.h5'.format(_ANGLE))
    with h5py.File('temp/data_D{:03d}.h5'.format(_ANGLE), 'a') as f:
        ds = f.create_dataset('input', data = indata)
        ds = f.create_dataset('target', data = target)
    return

def main():
    Start_Time = strftime("%Y-%m-%d_%H%M", localtime())
    cflist = glob.glob('{}/recog*/*wav'.format(clean_dir))
    sorted(cflist)
    with open('lista.txt', 'w+') as txtfl:
        txtfl.write('\n'.join(cflist))

    pool = Pool(processes=Pooling)
    angles = range(0,360,RESOLUTION)

    net = imp.load_source('Network', 'network.py')
    model = net.Network([CHANNELS_NET, OUTPUTS])
    if _GPU: model.to_gpu()
    
    _MODEL = sorted(glob.glob('train/*.model'))[NET]
    serializers.load_hdf5(_MODEL, model)
    model.train=False
    for _SNR in snr_vals:
        for fl in range(len(cflist)):
            _CLEAN = cflist[fl]
            _a = timeit.default_timer()
            print('Preparing Files... {}/{}'.format(fl+1,len(cflist)))
            pool.map(prepared_data, itertools.izip(angles, itertools.repeat(_CLEAN), itertools.repeat(_SNR)))
            print(timeit.default_timer()-_a)
            print("Forwarding Files...")
            for _ANGLE in angles:
                with h5py.File('temp/data_D{:03d}.h5'.format(_ANGLE)) as f:
                    _T = timeit.default_timer()
                    _outdata = model.forward(chainer.Variable(xp.asarray(f['input']), volatile='on'))    
                    if _GPU: 
                        _outdata = _outdata.data.get()
                    else:
                        _outdata = _outdata.data
                    _T = timeit.default_timer()-_T
                    argmax_angle = np.argmax(_outdata, axis=1)
                    predictd_angle = np.median(argmax_angle)
                    _snr = str(_SNR).replace('-', 'm')
                    log_file = _CLEAN.replace(clean_dir, 'Results_{}/{}dB/D{:03d}'.format(typeNoise, _snr, _ANGLE))
                    log_file = log_file.replace('wav', 'h5')
                    if os.path.exists(log_file):
                        os.remove(log_file)
                    outdir, _ = os.path.split(log_file) 
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)
                    with h5py.File(log_file,'a') as s:
                        dset = s.create_dataset("real", data=f['target'])
                        dset = s.create_dataset("predict", data=_outdata)
                        dset = s.create_dataset("time", data=_T)
                    print('SNR: {} - Predicted: {:06.2f} - True: {:06.2f} time:{:6.2f}secs'.format(_SNR, predictd_angle, _ANGLE, _T))
    Stop_time = strftime("%Y-%m-%d_%H%M", localtime())
    print('Forward Finished; Started at {} - Finished at {}'.format(Start_Time,Stop_time))

if __name__=='__main__':
    Pooling = 8
    current_exp = args.exp_date
    train_folder = 'training_files/{}'.format(current_exp)
    os.system('cp test.cfg ./{}/test.cfg'.format(train_folder))
    os.system('cp pyhark.py ./{}/pyhark.py'.format(train_folder))
    os.system('cp fft_generator.n ./{}/fft_generator.n'.format(train_folder))
    os.chdir(train_folder)
    print('Evaluate trained model for Sound Source Localization...')
    print('Running inside {}'.format(os.getcwd()))
    config_file='test.cfg'
    config.readfp(open(config_file))
    config.read(config_file)

    if config.getboolean('gpu', 'use'):
        cuda.get_device(config.getint('gpu', 'index')).use()
        xp = cuda.cupy
        _GPU = True
    else:
        xp = np
        _GPU = False
    datefl = current_exp
    skip = config.getint('audio', 'skip')
    frames = config.getint('audio', 'frames')
    snr_vals = config.get('audio', 'snr_vals')
    snr_vals = snr_vals.split(';')
    snr_vals = [ x for x in snr_vals]
    clean_dir = config.get('audio', 'clean_dir') 
    try:
        _MICS = config.get('audio', 'mics')
        _MICS = _MICS.split(';')
        _MICS = [ x for x in _MICS]
        _MICS = np.asarray(_MICS, dtype=np.int32)
        _DIFF = True
    except:
        _DIFF = False
    

    CHANNELS_SND = config.getint('audio', 'channels')
    CHANNELS_NET = config.getint('network', 'channels')
    OUTPUTS = config.getint('network', 'outputs')
    RESOLUTION = config.getint('angles', 'resolution')
    NET = config.getint('network', 'index')
    IMP_PREFIX = config.get('angles', 'impulses_prefix') 

    if not os.path.exists('temp'):
        os.makedirs('temp')
    if not os.path.exists('Results_{}'.format(typeNoise)):
        os.makedirs('Results_{}'.format(typeNoise))

    freeze_support()
    main()

    shutil.rmtree('temp')
