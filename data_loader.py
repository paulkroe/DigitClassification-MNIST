import pickle
import gzip
import os
import numpy as np

def load_data():
    "returns data and labels as np.arrays"
    assert(os.path.isfile("data\mnist.pkl.gz"))
    with gzip.open('data\mnist.pkl.gz', 'rb') as f:
        train_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    
    train_data = list(train_data)
    validation_data = list(validation_data)
    test_data = list(test_data)

    train_data[1] = vectorize(train_data[1])
    validation_data[1] = vectorize(validation_data[1])
    test_data[1] = vectorize(test_data[1])

    return train_data, validation_data, test_data

import numpy as np
def vectorize(int_array: np.array)->np.array:
    return np.array([np.array(make_zeros(i)) for i in int_array])

def make_zeros(i):
    temp = np.zeros(10)
    temp[i] = 1
    return temp