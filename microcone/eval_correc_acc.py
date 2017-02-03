#!/usr/bin/python
import os, glob, h5py
import numpy as np

base_folder = './testing_files/2016-07-14_1220/Results_SameNoise'

for snr in snr_vals:
    _snr = str(snr).replace('-', 'm')
    files = glob.glob('{}/{}dB/**/**/*h5'.format(base_folder, _snr))
    Results = []
    true_angle = np.asarray([], dtype = np.float32)
    full_angle = np.asarray([], dtype = np.float32)
    arg_angle = np.asarray([], dtype = np.float32)
    for archivo in files:
        with h5py.File(archivo, 'r') as f:
            _true_angle = np.zeros(f['real'].shape, dtype = np.int32)
            np.copyto(_true_angle, f['real'])
            outdata =  np.zeros(f['predict'].shape, dtype = np.float32)
            np.copyto(outdata, f['predict'])
        argmax = np.argmax(outdata, axis=1).astype(np.int32)
        _N, _S, _D, _I = [0, 0, 0, 0]
        _N = _true_angle.shape[0]
        for i in range(_true_angle.shape[0]):
            if (_true_angle[i] != 360) and (argmax[i] != 360):
                if not (np.abs(argmax[i] - _true_angle[i]) <= 20.):
                    _S += 1 # Incorrect Source Location
            elif (_true_angle[i] != 360) and (argmax[i] == 360):
                _D += 1 # MissDetection
            else:
                _I +=1  # Incorrect Insertion
        tmp_N = _N if _N >0 else 1         
        Correct = ((_N - _D - _S) / float(tmp_N)) * 100.0
        Acc = ((_N - _D - _S - _I) / float(tmp_N)) * 100.0
        print('N: {}, S: {}, D: {}, I: {} - C: {:.02f}, A: {:.02f}'.format(_N, _S, _D, _I, Correct, Acc))
        Results.append([_N, _S, _D, _I, Correct, Acc])
    savefile = 'correct_acc_{}.h5'.format(snr).replace('-', 'm')
    savefile = '{}/{}'.format(base_folder, savefile)
    if os.path.exists(savefile):
        os.remove(savefile)
    with h5py.File(savefile,'a') as f:
        dset = f.create_dataset("Results", data=Results)
print('Finish')
