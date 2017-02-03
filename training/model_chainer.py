from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import optimizers, serializers
from chainer.optimizer import GradientNoise

class Model(object):
    #TODO: Try to program a more general Model Class, even for reinforcement of Recurrent Networks 
    def __init__(self, config, Network):
        InitOpt= config.get('network', 'init_opt')
        InitOpt= [int(x) for x in InitOpt.split(';')] 


        if config.getboolean('gpu', 'use'):
            list_gpu = config.get('gpu', 'index')
            print('Configuring the training for GPU calculation:')
            print('  using gpus: {}'.format(list_gpu))
            self.list_gpu = [int(x) for x in list_gpu.split(';')  ]
            chainer.cuda.get_device(self.list_gpu[0]).use()
            self.xp = chainer.cuda
            self.Networks = [Network(InitOpt)] 
            
        else:
            print('Configuring the training for CPU calculation:')
            self.xp = np
            self.list_gpu = []
            self.Networks[0].train = True  
        self.Optimizer = optimizers.Adam(alpha=config.getfloat('train', 'learning_rate')) #TODO: Set type of Optimizer on Config File
        _inputs = config.get('data', 'labels')
        _inputs = [ x for x in _inputs.split(';')]
        self._inputs = len( _inputs)
        self._gaussian = config.getboolean('train', 'gaussian')
        if self._gaussian: self.eta = config.getfloat('train', 'eta_gn')
        self._lasso = config.getboolean('train', 'lasso')
        if self._lasso: self.lasso_dy = config.getfloat('train', 'decay_lasso')
        try: #only set on Recurrent Network
            self.sequence = config.getint('data', 'sequence')
            self.clip_threshold = config.getfloat('train', 'clip_threshold')
            self._use_clip = config.getboolean('train', 'use_clip')
            self._lstm = True 
            print('  Setting Network for Sequential Training...') 
        except:
            self._use_clip = False
            self._lstm = False
        self.train = False
        return

    def __call__(self, data, volatile='off'):
        """ Call Routine 
            Use only on training phase, for testing use Network.forward($VARIABLE_NAME)    
        """
        if not self.train:
            raise ImportError("Call only on training...")
        self.clear()
        if len(self.list_gpu) > 0:
            self.losses = [None] * len(self.list_gpu)
            for i in range(len(self.list_gpu)):
                self.forward_one_gpu(data[i], i, volatile)
        else:  
            data = [None]* self._inputs
            for i in range(self._inputs):
                data[i] = chainer.Variable(self.xp.asarray(data[0][i]), volatile=volatile)
            self.losses = self.Network[0](data[i])
        self.loss = self.losses[0] 
        self.accuracy =  self.Networks[0].accuracy   
        return self.loss

    def forward_one_gpu(self, batchdata, gpu, volatile):
        if self._lstm:
            losses = 0 
            self.Networks[gpu].reset_state()
            for i in range(self.sequence): 
                data = [None]* self._inputs
                for j in range(self._inputs):
                    data[j] = chainer.Variable(self.xp.to_gpu(batchdata[j][:,i], self.list_gpu[gpu]), volatile=volatile) 
                if i >0 :
                    data[-1] = context
                losses +=self.Networks[gpu](data)
                context = self.Networks[gpu].y
            self.losses[gpu] = losses 
        else:
            data = [None]* self._inputs
            for i in range(self._inputs):
                data[i] = chainer.Variable(self.xp.to_gpu(batchdata[i], self.list_gpu[gpu]), volatile=volatile) 
            self.losses[gpu] = self.Networks[gpu](data)        
        return

    def clear(self):
        self.loss = None
        self.losses = None
        self.accuracy = None
        return

    def use_gpu(self):
        if len(self.list_gpu) > 1:
            for i in range(1,len(self.list_gpu)):
                print('  Copying Network to GPU: {}'.format(self.list_gpu[i])) 
                self.Networks[i] = self.Networks[0].copy()
                self.Networks[i].to_gpu(self.list_gpu[i]) 
        print('  Moving Network to GPU: {}'.format(self.list_gpu[0]))
        self.Networks[0].to_gpu(self.list_gpu[0])
        return

    def SetUp(self):
        if len(self.list_gpu) > 0:
            self.use_gpu()
        self.Optimizer.setup(self.Networks[0])
        self.Networks[0].train = True
        if self._gaussian: 
            print('  Adding Gradient Noise Hook')
            self.Optimizer.add_hook(GradientNoise(self.eta))
        if self._use_clip:
            print('  Adding Gradient Clipping Hook')
            self.Optimizer.add_hook(chainer.optimizer.GradientClipping(self.clip_threshold))
        if self._lasso: 
            print('  Adding LASSO Regulation Hook')
            self.Optimizer.add_hook(chainer.optimizer.Lasso(self.lasso_dy))
        return

    def ZeroGrads(self):
        if len(self.list_gpu) > 0:
            for i in range(len(self.list_gpu)):
                self.Networks[i].cleargrads()
                #self.Networks[i].zerograds()
        else: 
            self.Networks[0].cleargrads()
            #self.Networks[i].zerograds()
        return 

    def BackWard(self):
        if len(self.list_gpu) > 0:
            for i in range(len(self.list_gpu)):
                self.losses[i].backward() 
        else: 
            self.losses[0].backward()
        return   

    def UpDate(self):
        if len(self.list_gpu) > 0:
            for i in range(1,len(self.list_gpu)):
                self.Networks[0].addgrads(self.Networks[i])
            self.Optimizer.update()
            for i in range(1,len(self.list_gpu)):
                self.Networks[i].copyparams(self.Networks[0])
        return

    def save(self, folder, epoch, batch):
        print('-'*5 , 'saving model')
        serializers.save_hdf5('{}/network_epoch{}_batch{}.model'.format(folder, epoch, batch), self.Networks[0])
        print('-'*5 , 'saving optimizer')
        serializers.save_hdf5('{}/network_epoch{}_batch{}.state'.format(folder, epoch, batch), self.Optimizer)
        return

    def LoadFineTnModel(self, folder, epoch, batch):
        print('Loading model')
        serializers.load_hdf5('{}/network_epoch{}_batch{}.model'.format(folder, epoch, batch), self.Networks[0])
        self.Networks[0].finetune_network()
        return

    def LoadResumeModel(self, folder, epoch, batch):
        print('Loading model')
        serializers.load_hdf5('{}/network_epoch{}_batch{}.model'.format(folder, epoch, batch), self.Networks[0])
        return

    def LoadTraining(self, folder, epoch, batch):
        print('Loading optimizer')
        serializers.load_hdf5('{}/network_epoch{}_batch{}.state'.format(folder, epoch, batch), self.Optimizer)
        return
