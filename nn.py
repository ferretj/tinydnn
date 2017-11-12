from copy import copy
import numpy as np
from batch_gen import BatchGen
from utils import softmax, logloss, dot12, outer12, flash


class DNN(object):

    def __init__(self, n_dense, neurons, n_in, n_softmax,
                 activation='relu', optimizer='sgd', lr=0.1,
                 weight_init='xav', bias_init='zero', loss='crossentropy'):
        self.n_dense = n_dense
        self.neurons = neurons
        self.n_in = n_in
        self.n_softmax = n_softmax
        self.optimizer = Optimizer.from_str(optimizer)
        self.activation = activation
        self.lr = lr
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.loss = Loss.from_str(loss)
        self.layers = self._layers()
        self._curr = 0

    def __getitem__(self, i):
        return self.layers[i]

    def __call__(self, X):
        return self._feedforward(X)

    def __iter__(self):
        self._curr = 0
        return self

    def next(self):
        if self._curr > self.n_dense - 1:
            raise StopIteration
        else:
            self._curr += 1
            return self.layers[self._curr - 1]

    def _layers(self):
        layers = []
        if self.n_dense > 1:
            if type(self.neurons) is int:
                layers.append(DLayer(self.n_in, self.neurons, self.activation, self.weight_init, self.bias_init))
                #layers.append(Activation.from_str(self.activation))
                for _ in xrange(1, self.n_dense - 1):
                    layers.append(DLayer(layers[-1], self.neurons, self.activation, self.weight_init, self.bias_init))
                    #layers.append(Activation.from_str(self.activation))
            elif type(self.neurons) is list:
                if len(self.neurons) == self.n_dense - 1 and type(self.neurons[0]) is int:
                    layers.append(DLayer(self.n_in, self.neurons[0], self.activation, self.weight_init, self.bias_init))
                    #layers.append(Activation.from_str(self.activation))
                    for ns in neurons[1:]:
                        layers.append(DLayer(layers[-1], ns, self.activation, self.weight_init, self.bias_init))
                        #layers.append(Activation.from_str(self.activation))
                else:
                    raise ValueError('Nope.')
            else:
                raise ValueError('Nope.')
            layers.append(DLayer(layers[-1], self.n_softmax, 'softmax', self.weight_init, self.bias_init))
        else:
            layers.append(DLayer(self.n_in, self.n_softmax, 'softmax', self.weight_init, self.bias_init))
        #layers.append(Activation.from_str('softmax'))
        return layers

    def _feedforward(self, X):
        self.X = X
        Xc = copy(X)
        for layer in self:
            Xc = layer(Xc)
            #if type(layer) is DLayer:
            #    print 'dense output : ', Xc
        self.output = Xc
        return self.output

    def _backprop(self, true, debug=False, return_grad=False):
        if debug:
            print 'Output shape : {}'.format(self.output.shape)
            print 'True output shape : {}'.format(true.shape)
        loss, grad = self.loss(self.output, true, output_grad=True)
        if debug:
            print 'loss backprop OK !'
        for i, layer in enumerate(self.layers[::-1]):
            if debug:
                print 'Layer : {}'.format(type(layer))
            lgrad = layer._backprop()
            if debug:
                print 'layer {} from end -> backprop OK !'.format(i + 1)
            layer._update(grad, self.optimizer, self.lr)
            if debug:
                print 'layer {} from end -> update OK !'.format(i + 1)
            grad = dot12(lgrad, grad)
            if debug:
                print 'layer {} from end -> tensordot OK !'.format(i + 1)
        if return_grad:
            return grad

    def train(self, X, y, batch_size=32, n_epochs=20, shuffle=True):
        for i in xrange(n_epochs):
            print 'Epoch {} / {}'.format(i + 1, n_epochs)
            batch_gen = BatchGen(X, y, batch_size, shuffle=shuffle)
            for j, (Xb, yb) in enumerate(batch_gen):
                flash('Batch', j, batch_gen.n_batches)
                _ = self._feedforward(Xb)
                self._backprop(yb)

    def apply(self, X, batch_size=32):
        r = []
        batch_gen = BatchGen(X, None, batch_size, shuffle=False)
        for Xb in batch_gen:
            rb = self._feedforward(Xb)
            r.append(rb)
        return np.vstack(r)


class DLayer(object):

    def __init__(self, prev, ncurr, activation, weight_init, bias_init, W=None, b=None):
        self.prev = prev
        self.ncurr = ncurr
        self.activation = Activation.from_str(activation)
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.W = W
        self.b = b
        if self.W is None:
            self._init_W()
        if self.b is None:
            self._init_b()

    @property 
    def nprev(self):
        if type(self.prev) is int:
            return self.prev
        elif type(self.prev) is np.ndarray:
            return self.prev.shape[1]
        elif type(self.prev) is DLayer: 
            return self.prev.ncurr
        else:
            raise TypeError('Wrong type for prev argument. Found {}'.format(type(self.prev)))

    def __call__(self, X):
        return self._feedforward(X)

    def _init_W(self):
        if self.weight_init == 'random':
            self.W = np.random.uniform(size=(self.nprev, self.ncurr))
        elif self.weight_init == 'xav':
            lim = np.sqrt(6. / (self.nprev + self.ncurr)) 
            self.W = np.random.uniform(low=-lim, high=lim, size=(self.nprev, self.ncurr))
        else:
            raise ValueError('Nope.')
        self.W = np.float32(self.W)

    def _init_b(self):
        if self.bias_init == 'zero':
            self.b = np.zeros((self.ncurr), dtype=np.float32)
        else:
            raise ValueError('Nope.')

    def _feedforward(self, X):
        self.X = X
        self.output = np.dot(X, self.W) + self.b
        print 'dense output : ', self.output 
        return self.activation(self.output)

    def _backprop(self):
        # calculate gradient of activation
        self.act_grad, self.act_bprop = self.activation._backprop()

        # calculate gradient wrt to input
        inp_grad = np.repeat(self.W[np.newaxis, :], self.X.shape[0], axis=0)
        if self.act_bprop == 'mul':
            inp_grad = inp_grad * self.act_grad[:, np.newaxis, :]
        else:
            inp_grad = dot12(inp_grad, self.act_grad)
        # calculate gradient wrt to parameters
        self.W_grad = np.zeros(self.X.shape + (self.W.shape[1],) + (self.W.shape[1],), dtype=np.float32)
        for i in xrange(self.W.shape[1]):
            try:
                self.W_grad[:, :, i, i] = self.X
            except IndexError:
                print 'i : {}'.format(i)
                print 'W shape : {}'.format(self.W.shape)
                print 'W_grad shape : {}'.format(self.W_grad.shape)
                print 'X shape : {}'.format(self.X.shape)
                raise
        self.b_grad = np.zeros((self.X.shape[0], self.b.shape[0], self.b.shape[0]), dtype=np.float32)
        inds_diag = np.arange(self.b.shape[0])
        try:
            self.b_grad[:, inds_diag, inds_diag] = np.ones(len(inds_diag), dtype=np.float32)
        except IndexError:
            print 'i : {}'.format(i)
            print 'b shape : {}'.format(self.b.shape)
            print 'b_grad shape : {}'.format(self.b_grad.shape)
            print 'X shape : {}'.format(self.X.shape)
            raise
        return inp_grad

    def _update(self, grad, optimizer, lr):
        self.X = None
        self.output = None
        #print len(grad)
        #print grad[0]
        #print 'zoulou'
        #print grad[1]
        #print type(grad)
        #print type(self.act_grad)
        if self.act_bprop == 'dot':
            grad = dot12(self.act_grad, grad)
            W_grad = dot12(self.W_grad, grad)
            b_grad = dot12(self.b_grad, grad)
        else:
            self.W_grad = self.W_grad * self.act_grad[:, np.newaxis, np.newaxis, :]
            self.b_grad = self.b_grad * self.act_grad[:, np.newaxis, :]
            W_grad = dot12(self.W_grad, grad)
            b_grad = dot12(self.b_grad, grad)
        print 'w grad : ', np.mean(W_grad, axis=0) 
        print 'b grad : ', np.mean(b_grad, axis=0)
        try:
            self.W = self.W - np.mean(W_grad, axis=0) * optimizer(lr)
        except ValueError:
            print grad.shape
            print self.W_grad.shape
            print W_grad.shape
            print self.W.shape
            raise   
        self.b = self.b - np.mean(b_grad, axis=0) * optimizer(lr)
        self.W_grad = None
        self.b_grad = None
        self.act_grad = None
        self.act_bprop = None


class Activation(object):

    def __init__(self, _func, _funcname):
        self._func = _func
        self._funcname = _funcname

    def __call__(self, arr):
        return self._feedforward(arr)

    @classmethod
    def from_str(cls, s):
        if s == 'relu':
            return cls(lambda x: np.maximum(0, x), s)
        elif s == 'softmax':
            return cls(softmax, s)
        else:
            raise ValueError('Nope.')

    def _feedforward(self, X):
        self.X = X
        self.output = self._func(X)
        return self.output

    def _backprop(self):
        if self._funcname == 'relu':
            # grad = np.zeros((self.output.shape[0], self.output.shape[1], self.output.shape[1]))
            # inds_diag = np.arange(self.output.shape[1])
            # grad[:, inds_diag, inds_diag] = np.clip(np.ceil(self.X), 0., 1.)
            # return grad                                  # shape OK
            grad = np.clip(np.ceil(self.X), 0., 1.)
            return grad, 'mul'
        elif self._funcname == 'softmax':
            # outc_diag = np.zeros((self.output.shape[0], self.output.shape[1], self.output.shape[1]))
            # inds_diag = np.arange(self.output.shape[1])
            # outc_diag[:, inds_diag, inds_diag] = self.output
            # grad = outc_diag - self.output[:, np.newaxis, :] ** 2  # shape OK
            # return grad
            inds_diag = np.arange(self.output.shape[1])
            grad = -1. * outer12(self.output)
            grad[:, inds_diag, inds_diag] += self.output
            return grad, 'dot'
        else:
            raise ValueError('Nope.')

    def _update(self, grad, optimizer, lr):
        self.X = None
        self.output = None
        pass


class Loss(object):

    def __init__(self, _func, _funcname):
        self._func = _func
        self._funcname = _funcname

    def __call__(self, y, yt, output_grad=False):
        l = self._feedforward(y, yt)
        if output_grad:
            return l, self._backprop()
        return l 

    @classmethod
    def from_str(cls, s):
        if s == 'crossentropy':
            return cls(lambda x, y: logloss(x, y), s)
        else:
            raise ValueError('Nope.')

    def _feedforward(self, y, yt):
        self.y = y
        self.yt = yt
        self.output = self._func(self.y, self.yt)
        print 'loss : ', self.output
        return self.output

    def _backprop(self):
        if self._funcname == 'crossentropy':
            # grad = - (np.log(10.) / self.y.shape[0]) * (1. / self.y)  # shape OK
            N = self.y.shape[0]
            nc = self.y.shape[1]
            if self.yt.ndim == 1:
                mask = np.eye(nc)[self.yt]
            else:
                mask = self.yt
            print 'mask : ', mask
            grad = - (1. / (N * self.y))
            print 'loss grad wrt to softmax output : ', mask * grad
            return mask * grad
        else:
            raise ValueError('Nope.')

    def _update(self, grad, optimizer, lr):
        self.y = None
        self.yt = None
        self.output = None
        pass


class Optimizer(object):

    def __init__(self, _func, _funcname):
        self._func = _func
        self._funcname = _funcname

    def __call__(self, lr):
        lr = self._func(lr)
        return lr

    @classmethod
    def from_str(cls, s):
        if s == 'sgd':
            return cls(lambda lr: lr, s)
        else:
            raise ValueError('Nope.')

