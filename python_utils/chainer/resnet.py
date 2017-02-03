import math
import chainer
import chainer.functions as F 
import chainer.links as L

class BottleNeckA(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride=2, act=F.elu):
        w = math.sqrt(2)
        super(BottleNeckA, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, w, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, w, nobias=True),
            bn3=L.BatchNormalization(out_size),

            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, w, nobias=True),
            bn4=L.BatchNormalization(out_size),
        )
        self.act=act

    def __call__(self, x, train, finetune=False):
        h1 = self.act(self.bn1(self.conv1(x), test=not train, finetune=finetune))
        h1 = self.act(self.bn2(self.conv2(h1), test=not train, finetune=finetune))
        h1 = self.bn3(self.conv3(h1), test=not train, finetune=finetune)
        h2 = self.bn4(self.conv4(x), test=not train, finetune=finetune)

        return self.act(h1 + h2)


class BottleNeckB(chainer.Chain):
    def __init__(self, in_size, ch, act=F.elu):
        w = math.sqrt(2)
        super(BottleNeckB, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0, w, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, w, nobias=True),
            bn3=L.BatchNormalization(in_size),
        )
        self.act=act

    def __call__(self, x, train, finetune=False):
        h = self.act(self.bn1(self.conv1(x), test=not train, finetune=finetune))
        h = self.act(self.bn2(self.conv2(h), test=not train, finetune=finetune))
        h = self.bn3(self.conv3(h), test=not train, finetune=finetune)

        return self.act(h + x)


class Block(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=2, act=F.elu):
        super(Block, self).__init__()
        links = [('a', BottleNeckA(in_size, ch, out_size, stride, act))]
        for i in range(layer-1):
            links += [('b{}'.format(i+1), BottleNeckB(out_size, ch, act))]

        for link in links:
            self.add_link(*link)
        self.forward = links

    def __call__(self, x, train, finetune=False):
        for name,_ in self.forward:
            f = getattr(self, name)
            h = f(x if name == 'a' else h, train, finetune)

        return h
