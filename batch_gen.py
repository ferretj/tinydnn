import numpy as np


class BatchGen(object):

    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._curr = 0
        self._limit = int(float(len(X)) / batch_size)

    def __iter__(self):
        return self

    def next(self):
        if self._curr > self._limit:
            raise StopIteration
        else:
            self._curr += 1
            if self.shuffle:
                inds = np.random.randint(len(X), size=(self.batch_size,))
                return self.X[inds]
            else:
                return self.X[self.batch_size * (self._curr - 1): self.batch_size * self._curr]
