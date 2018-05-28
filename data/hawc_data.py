"""
DataLoader for HAWC data (40x40 mapped)
Download Location: /home/danny/HAWC/data/gamma_image_mapping_data.npy
"""

import os
import sys
import numpy as np


def unpickle(file, labels):
    # fo = open(file, 'rb')
    # if (sys.version_info >= (3, 0)):
    #     import pickle
    #     d = pickle.load(fo, encoding='latin1')
    # else:
    #     import cPickle
    #     d = cPickle.load(fo)
    # fo.close()
    # return {'x': d['data'].reshape((10000, 3, 32, 32)), 'y': np.array(d['labels']).astype(np.uint8)}
    return {'x': np.load(file), 'y': np.load(labels)}


def load(data_dir, subset='train'):
    if subset == 'train':
        train_data = [unpickle(os.path.join(data_dir, 'gamma_image_mapping_data.npy'),
                               os.path.join(data_dir, 'gamma_labels.npy'))]
        trainx = np.concatenate([d['x'] for d in train_data], axis=0)
        trainy = np.concatenate([d['y'] for d in train_data], axis=0)
        return trainx, trainy
    elif subset == 'test':
        test_data = unpickle(os.path.join(data_dir, 'gamma_test_image_mapping_data.npy'),
                             os.path.join(data_dir, 'gamma_test_labels.npy'))
        testx = test_data['x']
        testy = test_data['y']
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')


class DataLoader(object):
    """ an object that generates batches of HAWC data for training
        each data point has the shape (N, 40, 40, 1), where we have just log-charge
        - support soon for (N, 40, 40, 2), where we have log-charge and hit time
        the label has the shape (N, 4), composed of the parameters:
            {azimuth, ...}
    """

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False):
        """
        - data_dir is location where to store files
        - subset is train|test
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels

        # create temporary storage for the data, if not yet created
        if not os.path.exists(data_dir):
            assert False, 'missing data folder'

        # load CIFAR-10 training data to RAM
        self.data, self.labels = load(data_dir, subset=subset)
        print('data shape:', self.data.shape)
        # self.data = np.transpose(self.data, (0, 2, 3, 1))  # (N,3,32,32) -> (N,32,32,3)

        self.p = 0  # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def get_num_labels(self):
        assert False
        return len(self.labels[0]) #np.amax(self.labels) + 1

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]
            self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset()  # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p: self.p + n]
        y = self.labels[self.p: self.p + n]
        self.p += self.batch_size

        if self.return_labels:
            return x, y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)


