import math
import chainer
import chainer.functions as F 
import chainer.links as L
from python_utils.chainer import Block

class Network(chainer.Chain):
    def __init__(self, InitOpt):
        w = math.sqrt(2) 
        super(Network, self).__init__(
            # Array decorrelator #De-Reverbarator
            faconv1 = L.Convolution2D(InitOpt[0],32,ksize=(1,1), nobias=True),
            fabn1 = L.BatchNormalization(32),
            faconv2 = L.Convolution2D(32,32,ksize=(1,1), nobias=True),
            fabn2 = L.BatchNormalization(32),
            fares1 = Block(3, 32, 32, 32, 1),
            # Audio Locator       
            conv1 = L.Convolution2D(32,32,ksize=(45,4), nobias=True),
            bn1 = L.BatchNormalization(32),
            conv2 = L.Convolution2D(32,32,ksize=(45,4), nobias=True),
            bn2 = L.BatchNormalization(32),
            conv3 = L.Convolution2D(32,64,ksize=(45,4), nobias=True),
            bn3 = L.BatchNormalization(64),
            conv4 = L.Convolution2D(64,128,ksize=(45,4), nobias=True),
            bn4 = L.BatchNormalization(128),
            conv5 = L.Convolution2D(128,256,ksize=(45,4), nobias=True),
            bn5 = L.BatchNormalization(256),
            conv6 = L.Convolution2D(256,4096,ksize=(28,2), nobias=True),
            bn6 = L.BatchNormalization(4096),
            conv7 = L.Convolution2D(512,4096,ksize=1, nobias=True),
            bn7 = L.BatchNormalization(4096),
            ip1 = L.Linear(4096, InitOpt[1]),

        )
        self.train=True
        self.InitOpt = InitOpt
        self.act = F.elu

 
    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, variables):
        self.clear()   
        y = self.encode(variables[0])
        self.loss = F.softmax_cross_entropy(y,variables[1])
        self.accuracy = F.accuracy(y,variables[1])
        return self.loss

    def forward(self,h):
        return F.softmax(self.encode(h))

    def encode(self,h,printing = False):
        """ Array decorrelator"""
        if printing: print 'input', h.data.shape
        h = self.act(self.fabn1(self.faconv1(h), test=not self.train))
        h = self.act(self.fabn2(self.faconv2(h), test=not self.train))
        h = self.fares1(h, self.train)
        if printing: print 'res1', h.data.shape
        """ Audio Locator """
        h = self.act(self.bn1(self.conv1(h), test=not self.train))
        if printing: print 'conv1',h.data.shape
        h = F.max_pooling_2d(h,ksize=(1,4),stride=1)
        if printing: print 'pool1',h.data.shape
        h = self.act(self.bn2(self.conv2(h), test=not self.train))
        if printing: print 'conv2',h.data.shape
        h = self.act(self.bn3(self.conv3(h), test=not self.train))
        if printing: print 'conv3',h.data.shape
        h = F.max_pooling_2d(h,ksize=(4,1),stride=1)
        if printing: print 'pool2',h.data.shape
        h = self.act(self.bn4(self.conv4(h), test=not self.train))
        if printing: print 'conv4',h.data.shape
        h = self.act(self.bn5(self.conv5(h), test=not self.train))
        if printing: print 'conv5',h.data.shape
        h = self.act(self.bn6(self.conv6(h), test=not self.train))
        h = self.act(self.bn7(self.conv7(h), test=not self.train))
        if printing: print 'conv7',h.data.shape
        h = self.ip1(h)
        return h
