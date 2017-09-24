import numpy as np
from numpy.core.umath_tests import matrix_multiply

def softmax(x, eps=1e-12):
    expo = np.exp(x)
    return expo / (np.sum(expo) + eps)

def logloss(x, y):
    if y.ndim == 1:
        n_classes = len(np.unique(y))
        y = np.eye(n_classes)[y]
    logx = np.log(np.clip(x, 0., 1.))
    return - np.mean(np.sum(np.multiply(logx, y), axis=1))

def dot12(a, b):
    if a.ndim == b.ndim == 3:
        return (a[..., np.newaxis] * b[:, np.newaxis, ...]).sum(axis=-2)
    elif a.ndim == 3 and b.ndim == 2:
        try:
            return matrix_multiply(a, b[..., np.newaxis])[:, :, 0]
        except ValueError:
            print 'Left shape : {}'.format(a.shape)
            print 'Right shape : {}'.format(b.shape)
            raise
    elif a.ndim == 4 and b.ndim == 2:
        return matrix_multiply(a, b[:, np.newaxis, :, np.newaxis])[:, :, :, 0]
    else:
        raise ValueError('Dimensional mismatch between a (shape: {}) and b (shape: {})'.format(a.shape, b.shape))
