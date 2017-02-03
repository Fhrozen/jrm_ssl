from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, h5py, glob
import numpy as np
from matplotlib import pyplot as plt

class hdf5_plain(object):
    def __init__(self, config_file):
        self.prefix = config_file.get('database', 'prefix')
        _inputs = config_file.get('data', 'labels')
        self._inputs = [ x for x in _inputs.split(';')]  
        self.epochs_ = config_file.getint('train', 'epochs')
        self.get_db_sizes()    

    def get_db_sizes(self):
        print('Looking at {} for files:'.format(self.prefix))
        list_dirs = glob.glob('{}/*'.format(self.prefix))
        list_file = []
        index = []
        first = True
        for i in range(len(list_dirs)):
            _files = glob.glob('{}/*'.format(list_dirs[i]))
            if first: 
                first = False
                with h5py.File(_files[0], 'r') as f:
                    _dims = [ None ] * len(self._inputs)
                    _types = [ None ] * len(self._inputs)
                    for j in range(len(self._inputs)):
                        testfile= f[self._inputs[j]]
                        _dim = testfile.shape
                        _dims[j] = _dim if len(_dim) !=0 else []
                        _types[j] = testfile.dtype
                        print('  data label:{} dim:{} dtype:{}'.format(self._inputs[j], list(_dims[j]), _types[j]))    
            list_file += [_files]           
            index += [[i, x] for x in np.arange(len(_files))] 
        self._dims = _dims
        self._type = _types  
        self.files = list_file 
        self.idxs = index 
        print('  Total of {} files on {} folders...'.format(len(self.idxs), len(self.files)))
        return

    def read_data(self, idxs, divisions):
        """
            Data feeder from HDF5 Files
            idxs[,0]: Folder
            indx[,1]: Filename
        """
        data_batch = np.empty((divisions),dtype=object) 
        idx_div = np.linspace(0, len(idxs), num = divisions + 1, dtype=np.int)
        for i in range(divisions):
            batch_in = idx_div[i+1] - idx_div[i]
            _idxs = idxs[idx_div[i]:idx_div[i+1]]
            data_labels = np.empty((len(self._inputs)),dtype=object)
            for j in range(batch_in):
                iDb, iFL = _idxs[j]
                with h5py.File(self.files[iDb][iFL], 'r') as f:
                    for k in range(len(self._inputs)):
                        if j == 0:
                            dims = [batch_in]
                            dims +=  [ x for x in self._dims[k]]
                            data_labels[k] = np.zeros(dims, dtype=self._type[k])
                        data_labels[k][j] = np.asarray(f[self._inputs[k]])
            data_batch[i] = data_labels
        return data_batch

class hdf5_sigmoid(object):
    def __init__(self, config_file):
        self.prefix = config_file.get('database', 'prefix')
        _inputs = config_file.get('data', 'labels')
        self._inputs = [ x for x in _inputs.split(';')]  
        self.epochs_ = config_file.getint('train', 'epochs')
        self.get_db_sizes()    

    def get_db_sizes(self):
        print('Looking at {} for files:'.format(self.prefix))
        list_dirs = glob.glob('{}/*'.format(self.prefix))
        list_file = []
        index = []
        first = True
        for i in range(len(list_dirs)):
            _files = glob.glob('{}/*'.format(list_dirs[i]))
            if first: 
                first = False
                with h5py.File(_files[0], 'r') as f:
                    _dims = [ None ] * len(self._inputs)
                    _types = [ None ] * len(self._inputs)
                    for j in range(len(self._inputs)):
                        testfile= f[self._inputs[j]]
                        _dim = testfile.shape
                        _dims[j] = _dim if len(_dim) !=0 else []
                        _types[j] = testfile.dtype
                        print('  data label:{} dim:{} dtype:{}'.format(self._inputs[j], list(_dims[j]), _types[j]))    
            list_file += [_files]           
            index += [[i, x] for x in np.arange(len(_files))] 
        self._dims = _dims
        self._type = _types  
        self.files = list_file 
        self.idxs = index 
        print('  Total of {} files on {} folders...'.format(len(self.idxs), len(self.files)))
        return

    def read_data(self, idxs, divisions):
        """
            Data feeder from HDF5 Files
            idxs[,0]: Folder
            indx[,1]: Filename
        """
        data_batch = np.empty((divisions),dtype=object) 
        idx_div = np.linspace(0, len(idxs), num = divisions + 1, dtype=np.int)
        for i in range(divisions):
            batch_in = idx_div[i+1] - idx_div[i]
            _idxs = idxs[idx_div[i]:idx_div[i+1]]
            data_labels = np.empty((len(self._inputs)),dtype=object)
            for j in range(batch_in):
                iDb, iFL = _idxs[j]
                with h5py.File(self.files[iDb][iFL], 'r') as f:
                    for k in range(len(self._inputs)):
                        if j == 0:
                            dims = [batch_in]
                            dims +=  [ x for x in self._dims[k]]
                            data_labels[k] = np.zeros(dims, dtype=self._type[k])
                        #tmp = np.zeros(.shape, 
                        data_labels[k][j] = np.asarray(f[self._inputs[k]])
            data_batch[i] = data_labels
        return data_batch

class hdf5_new_segment(object):
    def __init__(self, config_file):
        self.prefix = config_file.get('database', 'prefix')
        _inputs = config_file.get('data', 'labels')
        self._inputs = [ x for x in _inputs.split(';')]  
        self.epochs_ = config_file.getint('train', 'epochs')
        self.get_db_sizes()    

    def get_db_sizes(self):
        print('Looking at {} for files:'.format(self.prefix))
        list_dirs = glob.glob('{}/*'.format(self.prefix))
        list_file = []
        index = []
        first = True
        for i in range(len(list_dirs)):
            _files = glob.glob('{}/*'.format(list_dirs[i]))
            if first: 
                first = False
                with h5py.File(_files[0], 'r') as f:
                    _dims = [ None ] * len(self._inputs)
                    _types = [ None ] * len(self._inputs)
                    for j in range(len(self._inputs)):
                        testfile= f[self._inputs[j]]
                        _dim = testfile.shape
                        _dims[j] = _dim if len(_dim) !=0 else []
                        _types[j] = testfile.dtype
                        print('  data label:{} dim:{} dtype:{}'.format(self._inputs[j], list(_dims[j]), _types[j]))    
            list_file += [_files]           
            index += [[i, x] for x in np.arange(len(_files))] 
        self._dims = _dims
        self._type = _types  
        self.files = list_file 
        self.idxs = index 
        print('  Total of {} files on {} folders...'.format(len(self.idxs), len(self.files)))
        return

    def read_data(self, idxs, divisions):
        """
            Data feeder from HDF5 Files
            idxs[,0]: Folder
            indx[,1]: Filename
        """
        data_batch = np.empty((divisions),dtype=object) 
        idx_div = np.linspace(0, len(idxs), num = divisions + 1, dtype=np.int)
        for i in range(divisions):
            batch_in = idx_div[i+1] - idx_div[i]
            _idxs = idxs[idx_div[i]:idx_div[i+1]]
            data_labels = np.empty((len(self._inputs)),dtype=object)
            for j in range(batch_in):
                iDb, iFL = _idxs[j]
                with h5py.File(self.files[iDb][iFL], 'r') as f:
                    for k in range(len(self._inputs)):
                        if j == 0:
                            dims = [batch_in]
                            if k == 0:
                                data_labels[k] = np.zeros((batch_in,1,257,7*20), dtype=self._type[k])
                            else:
                                dims +=  [ x for x in self._dims[k]]
                                data_labels[k] = np.zeros(dims, dtype=self._type[k])                                
                        if k == 0:
                            a = np.zeros((1,257,7*20), dtype=np.float32)
                            for _fr in range(20):
                                for _ch in range(7):
                                    a[0,:,_ch+_fr*7] = f[self._inputs[k]][_ch, :,_fr]
                            data_labels[k][j] = a
                        else:
                            data_labels[k][j] = np.asarray(f[self._inputs[k]])
            data_batch[i] = data_labels
        return data_batch