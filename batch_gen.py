import numpy as np


class BatchGen(object):

    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = int(float(len(X)) / batch_size)
        self._curr = 0

    def __iter__(self):
        return self

    def next(self):
        if self._curr > self.n_batches:
            raise StopIteration
        else:
            self._curr += 1
            if self.shuffle:
                inds = np.random.randint(len(self.X), size=(self.batch_size,))
                if self.y is not None:
                    return self.X[inds], self.y[inds]
                return self.X[inds]
            else:
                sl = slice(self.batch_size * (self._curr - 1), self.batch_size * self._curr)
                if self.y is not None:
                    return self.X[sl], self.y[sl]
                return self.X[sl]
