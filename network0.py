import numpy as np
seed_value = 42

# TODO: assert input right shape for nn

class network:
    def __init__(self, layers: np.array, seed_value: float = None):
        
        '''
        Implementing a fully connected neural network.
        The network consists of two np.arrays, namely weights and biases.
        weights[l] = weights matrix of the l+1-th layer. --> weights[l][j][i] = w_{ji}_{l+1}
        biases[l] = biases of the l+1-th layer. --> biases[l][i] = b^{l+1}_{i}
        '''
        self.layers = layers
        np.random.seed(seed_value)
        self.weights = [np.random.rand(layers[i+1], layers[i]) for i in range(len(layers)-1)]
        self.biases = [np.random.rand(layers[i],1) for i in range(1, len(layers))]

    def forward(self, z : np.array)->np.array:
        z = [z] # need to do this with a list, since not all sublist are of the same size
        a = z[:] # create copy of z

        z.append(np.dot(self.weights[0], a[0]) + self.biases[0][0])        
        a.append(ReLU(z[-1]))


        for i in range(1,len(self.layers)-1):
            z.append(np.dot(self.weights[i], a[-1]) + self.biases[i][0])
            a.append(ReLU(z[-1]))
        return z , a # returns two lists
    
    def SGD(self, train_data: np.array, eta: float, epochs: int, batch_size: int, loss_fn, dloss_fn, report: bool = False, validation_data: np.array = None, seed_value: float = None):
        '''if report is true, then we need validation data'''
        data = train_data
        np.random.seed(seed_value)
        for epoch in range(epochs):
            np.random.shuffle(data) # in place
            for i in range(0,len(data), batch_size):
                self.batch_update(data[i:i+batch_size], eta, loss_fn=loss_fn, dloss_fn=dloss_fn)

            if report:
                # might be more interesting to calculate loss and accuracy after each batch update, not after each epoch
                loss, accuracy = 0, 0

                for (X,y) in validation_data:
                    y_pred = self.forward(X)[-1]
                    loss += loss_fn(y_pred=y_pred, y_true= y)
                    # this should be done better
                    temp = np.zeros_like(y_pred)
                    temp[np.argmax(y_pred)] = 1
                    accuracy += np.array_equal(temp, y)

                loss /= len(validation_data)
                accuracy /= len(validation_data)
                print(f"epoch: {epoch} | loss: {loss} | accuracy: {accuracy}")


    def batch_update(self, train_data: np.array, eta: float, loss_fn, dloss_fn):
        partial_weights = np.zeros_like(self.weights)
        partial_biases = np.zeros_like(self.biases)

        for (X,y) in train_data:
            weights_update, biases_update  = self.backpropagation(X, y, dloss_fn)
            partial_weights += weights_update
            partial_biases += biases_update
        
        partial_weights = [(-eta*sublist)/len(train_data) for sublist in partial_weights] # normalize and multiply by learning rate
        partial_biases = [(-eta*sublist)/len(train_data) for sublist in partial_biases]


        self.weights = [np.add(x,y) for (x,y) in zip(self.weights, partial_weights)]
        self.biases  = [np.add(x,y) for (x,y) in zip(self.biases, partial_biases)]



    def backpropagation(self, X: np.array, y: np.array, dloss_fn):
        z, a = self.forward(X)
        y = np.array(y).reshape(-1,1)

        

        delta = z[1:] # so that delta is initialized with the correct size
        delta.reverse() # error in the last layer is in the first entry of delta, etc.
        delta[0] = dloss_fn(y_true=y, y_pred=a[-1]) * dReLU(z[-1])
        print(f"Print delta: {delta}")
        print(f"Print weights: {self.weights}")
        print(f"Print activation: {a}")

        for l in range(1, len(delta)):
            delta[l] = (np.dot(np.transpose(self.weights[-l]), delta[l-1])) * dReLU(z[-(l+1)])
            
        delta.reverse()
        
        return self.weights, delta # partial_weights is the same as delta according to backpropagation formula 3)

def ReLU(input: np.array):
        return np.maximum(0,input)

def dReLU(input: np.array):
        return np.where(input>0, 1, 0)

# when using the network with ReLU one can't easily use cross entropy loss
def mean_square_error(y_true: np.array, y_pred: np.array):
    '''square mean error for one training input'''
    return np.mean(np.square(y_pred-y_true))
     

def dmean_square_error(y_true: np.array, y_pred: np.array):
    return 2/len(y_true) * (y_pred-y_true)

