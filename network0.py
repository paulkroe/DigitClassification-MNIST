import numpy as np
import random
from tqdm.auto import tqdm
seed_value = 42
random.seed(seed_value)

class network:
    def __init__(self, layers: np.array):
        
        '''
        Implementing a fully connected neural network.
        The network consists of two np.arrays, namely weights and biases.
        weights[l] = weights matrix of the l+1-th layer. -->weigths[l][j][i] = w_{ji}_{l+1}
        biases[l] = biases of the l+1-th layer. --> biases[l][i] = b^{l+1}_{i}
        '''
        self.layers = layers
        self.weigths = [np.random.rand(layers[i+1], layers[i]) for i in range(len(layers)-1)]
        self.biases = [np.random.rand(layers[i],1) for i in range(1, len(layers))]

    def forward(self, a : np.array)->np.array:
        for i in range(len(self.layers)-1):
            a = self.ReLU(np.dot(self.weigths[i], a) + np.transpose(self.biases[i][0])) # numpy will transponse a if necessary, but not b (since we want pointwise addition), added the zeros there since np will return an array of an array
        return a
    
    def SGD(self, train_data: np.array, eta: float, epochs: int, batch_size: int, loss_fn, report: bool = False, validation_data: np.array = None):
        '''if report is true, then we need validation data'''
        data = train_data

        for epoch in tqdm(range(epochs)):
            np.random.shuffle(data) # in place
            for i in range(0,len(data), batch_size):
                self.batch_update(data[i:i+batch_size], eta)

            if report:
                # might be more interesting to calculate loss and accuracy after each batch update, not after each epoch
                loss, accuracy = 0, 0

                for (X,y) in validation_data:
                    y_pred = self.forward(X)
                    loss += loss_fn(y_pred=y_pred, y_true= y)
                    # this should be done better
                    temp = np.zeros_like(y_pred)
                    temp[np.argmax(y_pred)] = 1
                    accuracy += np.array_equal(temp, y)

                loss /= len(validation_data)
                accuracy /= len(validation_data)


    def batch_update(self, train_data: np.array, eta: float):
        partial_weigths = np.zeros_like(self.weigths)
        partial_biases = np.zeros_like(self.biases)

        for (X,y) in train_data:
            weigths_update, biases_update  = self.backpropagation(X, y)
            partial_weigths += weigths_update
            partial_biases += biases_update
        
        partial_weigths /= len(train_data)
        partial_biases /= len(train_data)
        
        self.weigths -= eta * partial_weigths
        self.biases -= eta * partial_biases


    def backpropagation(self):
        pass

    def ReLU(self, input: np.array):
        return np.maximum(0,input)        

    def dReLU(self, input: np.array):
        return np.where(input>0, 1, 0)