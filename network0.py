import importlib
import numpy as np
import debugger as deb
importlib.reload(deb)
import json
import matplotlib.pyplot as plt

seed_value = 42

# TODO: assert input right shape for nn

class network:
    def __init__(self, layers: np.array, weights=None, biases=None, seed_value: float = None):
        
        '''
        Implementing a fully connected neural network.
        The network consists of two np.arrays, namely weights and biases.
        weights[l] = weights matrix of the l+1-th layer. --> weights[l][j][i] = w_{ji}_{l+1}
        biases[l] = biases of the l+1-th layer. --> biases[l][i] = b^{l+1}_{i}
        '''
        assert(isinstance(layers, np.ndarray))

        self.layers = layers
        
        if not (weights is None and biases is None):
            self.layers = layers
            self.weights = weights
            self.biases = biases 


        else:
            np.random.seed(seed_value)
            self.weights = [np.random.rand(layers[i+1], layers[i]) for i in range(len(layers)-1)]
            self.biases = [np.random.rand(layers[i]) for i in range(1, len(layers))]

        #deb.check_weights(self, self.weights)
        #deb.check_biases(self, self.biases)
    
    def forward(self, z : np.array, internal: bool = False):
        z = [z] # need to do this with a list, since not all sublist are of the same size
        a = z[:] # create copy of z

        z.append(np.dot(self.weights[0], a[0]) + self.biases[0])        
        a.append(ReLU(z[-1]))


        for i in range(1,len(self.layers)-1):
            z.append(np.dot(self.weights[i], a[-1]) + self.biases[i])
            a.append(ReLU(z[-1]))
        if internal:
            return z, a
        else:
            return a[-1]    
    def SGD(self, train_data: np.array, eta: float, epochs: int, batch_size: int, loss_fn, dloss_fn, report: bool = False, validation_data: np.array = None, seed_value: float = None):
        losses=[]
        '''if report is true, then we need validation data
        input data has the form [np.array(input),np.array(label)], due to initial form of the data'''
        # should shuffle data here
        assert(len(train_data[0]) == len(train_data[1]))
        n = len(train_data[0])
        for epoch in range(epochs):

            for i in range(0,n, batch_size):
                self.batch_update(train_data[0][i:i+batch_size], train_data[1][i:i+batch_size], eta, dloss_fn=dloss_fn)

            if report:
                assert(not (validation_data is None))
                assert(len(validation_data[0]) == len(validation_data[1]))
                # might be more interesting to calculate loss and accuracy after each batch update, not after each epoch
                loss = 0

                for i in range(len(validation_data)):
                    y_pred = self.forward(validation_data[0][i])
                    loss += loss_fn(y_pred=y_pred, y_true= validation_data[1][i])
                    # get y_pred in one-hot

                loss /= len(validation_data[0])
                losses.append(loss)
                print(f"epoch: {epoch} | loss: {loss}")
        plt.plot(losses, label="loss vs epochs")
        plt.legend()
        plt.show()

    def batch_update(self, train_data_feature: np.array, train_data_label: np.array, eta: float, dloss_fn):
        assert(len(train_data_feature) == len(train_data_label))
        n = len(train_data_feature)

        partial_weights = [np.zeros((self.layers[i+1], self.layers[i])) for i in range(len(self.layers)-1)]
        partial_biases = [np.zeros(self.layers[i]) for i in range(1, len(self.layers))]


        for i in range(n):
            assert(isinstance(train_data_feature[i], np.ndarray) and isinstance(train_data_label[i], np.ndarray))
            weights_update, biases_update  = self.backpropagation(train_data_feature[i], train_data_label[i], dloss_fn)
            partial_weights = [np.add(x,y) for (x,y) in zip(partial_weights, weights_update)]
            partial_biases = [np.add(x,y) for (x,y) in zip(partial_biases, biases_update)]

        partial_weights = [(-eta*sublist)/n for sublist in partial_weights] # normalize and multiply by learning rate
        partial_biases = [(-eta*sublist)/n for sublist in partial_biases]
        
        # print(f"partial weights: {partial_weights}")
        # print(f"weights: {self.weights}")
        # print(f"partial biases: {partial_biases}")
        # print(f"biases: {self.biases}") 
        # deb.check_biases(self, partial_weights)
        # deb.check_weights(self, partial_biases)

        self.weights = [np.add(x,y) for (x,y) in zip(self.weights, partial_weights)]
        self.biases  = [np.add(x,y) for (x,y) in zip(self.biases, partial_biases)]


    def backpropagation(self, X: np.array, y: np.array, dloss_fn):
        assert(isinstance(X, np.ndarray) and isinstance(y, np.ndarray))

        z, a = self.forward(X, internal=True)
        delta = []
        delta.append(dloss_fn(y_true=y, y_pred=a[-1])*dReLU(z[-1]))

        for i in range(1, len(self.layers)-1):
            delta.append(np.dot(np.transpose(self.weights[-i]), delta[-1])*z[-(i+1)])

        # calculate partial weights
        partial_weights = []

        for ind_delta in range(len(delta)):
            if(isinstance(delta[ind_delta], np.ndarray)): # check if array or scalar
                temp = []
                for d in range(len(delta[ind_delta])):
                    # print(f"print a in loop {a[-(ind_delta+2)]}")
                    temp.append(delta[ind_delta][d] * a[-(ind_delta+2)])
                partial_weights.append(np.array(temp))
            else:
                # print(f"print a in loop {a[-(ind_delta+2)]}")
                partial_weights.append(np.array(delta[ind_delta] * a[-(ind_delta+2)]))
        delta.reverse()
        partial_weights.reverse()
    
        # deb.check_biases(self, delta)
        # deb.check_weights(self, partial_weights)
        

        return partial_weights, delta # since delta = partial_biases

    def safe_model(self, file_name: str):
        w = self.weights[:]
        b = self.biases[:]

        w = w.tolist()
        b.tolist()

        data = {"weights": w, 
                "biases": b
                }
        with open(file_name, "w") as f:
            json.dump(data, f)


def ReLU(input: np.array):
        return np.maximum(0,input)

def dReLU(input: np.array):
        return np.where(input>0, 1, 0)

# when using the network with ReLU one can't easily use cross entropy loss
def mean_square_error(y_true: np.array, y_pred: np.array):
    '''square mean error for one training input'''
    #TODO: check that output has right format 
    return np.mean(np.square(y_pred-y_true))
     

def dmean_square_error(y_true: np.array, y_pred: np.array):
    if not np.issubdtype(y_true.dtype, float):
        return 2/len(y_true) * (y_pred-y_true)
    else:
        return 2*(y_pred-y_true)

