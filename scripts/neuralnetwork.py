import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PATH_TO_FASHION_TEST = "../data/fashion_test.npy"
PATH_TO_FASHION_TRAIN = "../data/fashion_train.npy"

def load_data_normalized(path):
    
    def one_hot(y):
        table = np.zeros((y.shape[0], 5))
        for i in range(y.shape[0]):
            table[i][int(y[i][0])] = 1 
        return table

    def normalize(x): 
        x = x / 255
        return x 

    data = np.load(path)
    X, y = np.hsplit(data, [-1])
    return normalize(X),one_hot(y), y


class NeuralNetwork:

    class Layer:

        def __init__(self, input_size, output_size, activation_function):
            # shape
            self._input_size = input_size
            self._output_size =  output_size
            # weights 
            self._ws = self.create_weights((input_size,output_size))
            # data for layer
            self._x = None
            self._z = None # Xw
            self._b = self.create_bias(self._ws.shape[1])
            self._a = None # sigma(Xw + b)
            # activation function
            self._activation_function = activation_function
            if activation_function == NeuralNetwork.ReLU or activation_function == NeuralNetwork.leaky_ReLU: 
                self._activation_function_der = NeuralNetwork.dReLU

        def create_weights(self, shape, initialization = "X"):
            """
            description
            -------
            Input layer:    
                Initialize in w in (p x k) 
                In this case p is 784
            Hidden layer:
                Initialize w in (k1 x k2)
            Output layer:
                Initialize w in (k_l x labels)
                In this case 5 labels
            Parameters
            ---------
            from_neurons: int
                Input layer: The p-features (in this case 784 + 1)
                Hidden, output layer: the number of neurons in privous layer.
            to_neurons: int

            Returns
            -----
            ws: list 
                A list of weights for a layer in the neural network
                in the dimension (n x k)
            """
            w = np.random.random(shape) # pick random numbers between 0-1 in from of shape
            if initialization == "X":
                w = w / np.sqrt(shape[0])
            o_o = np.random.choice([-1,1],shape,True) # to make half the number negative
            w = w * o_o # finalize the weights 
            return w
        
        def create_bias(self, size):
            """
            create bias
            """
            return np.random.randn(size,)


    def __init__(self, X, y, batch_size = 32, lr = 0.001,  epochs = 150):
        # data
        self.input = X 
        self.target = y

        self.batch = batch_size
        self.epochs = epochs
        self.lr = lr
        
        self.x = self.input[:self.batch] # batch input 
        self.y = self.target[:self.batch] # batch target value
        self.loss = []
        self.acc = []
        # layers
        self.L1 = self.Layer(self.input.shape[1],256, activation_function=self.ReLU)
        self.L1._x = self.x
        self.L2 = self.Layer(256,128, activation_function=self.ReLU)
        self.L3 = self.Layer(128,5, activation_function=self.softmax)
        self.layers = [] 

    def add_layer(self,input_size, output_size, activation_function):
        layer = self.Layer(input_size,output_size, activation_function)
        if len(self.layers) == 0:
            layer._x = self.x
        self.layers.append(layer)

    def init_layers(self):
        
        # create weights
        self.W1 = self.Layer.create_weights((self.input.shape[1], 256),initialization="X")
        self.W2 = self.Layer.create_weights((self.W1.shape[1],128),initialization="X")
        self.W3 = self.Layer.create_weights((self.W2.shape[1],self.y.shape[1]),initialization="X")
        # create bias
        self.b1 = self.Layer.create_bias(self.W1.shape[1])
        self.b2 = self.Layer.create_bias(self.W2.shape[1])
        self.b3 = self.Layer.create_bias(self.W3.shape[1])
    
    # activation functions
    def ReLU(self, x):
        return np.maximum(0,x)

    def dReLU(self,x):
        return 1 * (x > 0) 
    
    def leaky_ReLU(self, x):
            return np.maximum(0.05*x,x)
    
    def der_leaky_ReLU(self, x):
        return self.dReLU(x)
    
    def softmax(self, z):
        z = z - np.max(z, axis = 1).reshape(z.shape[0],1)
        return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0],1)
    
    # helper functions
    def shuffle(self):
        idx = [i for i in range(self.input.shape[0])]
        np.random.shuffle(idx)
        self.input = self.input[idx]
        self.target = self.target[idx]
    
    def _get_reversed_list(self, x):
        """
        returns the list backwards
        """
        r_list = list(reversed(x))
        return r_list

    def feedforward(self):
        self.layers[0]._x = self.x
        new_layer_list = []
        for i,layer in enumerate(self.layers):
            if len(new_layer_list) == 0: # if this is the first layer
                layer._z = layer._x.dot(layer._ws) + layer._b # z = Xw + b
            else: # other than first layer
                layer._x = new_layer_list[-1]._a
                layer._z = layer._x.dot(layer._ws) + layer._b # z = aw + b

            layer._a = layer._activation_function(layer._z) # a = sigma(z)
            # append layer to new layer list
            new_layer_list.append(layer)
        # error
        self.error = new_layer_list[-1]._a - self.y
        # new layers 
        self.layers = new_layer_list
    
        
    def backprop(self):
        r_layers = self._get_reversed_list(self.layers)
        dL_dz= (1/self.batch)*self.error*2
        for i, layer in enumerate(r_layers,start=1):
            if i == 1:
                # dL_dw = layer._x.T @ dL_dz.T 
                dL_dw = (dL_dz.T @ layer._x).T # dL_dz 
                dL_db = dL_dz.sum(axis=0)
            else:
                # weights
                dL_dah = (dL_dz_old) @ old_weight.T # dzo_dah = old_weight
                dah_dzh = (dL_dah * self.dReLU(layer._z)).T
                dL_dw = (dah_dzh @ layer._x).T
                # bias
                dL_dz = dL_dah * dah_dzh.T 
                dL_db = dL_dz.sum(axis=0)
            
            # to use for next iteration
            old_weight = layer._ws
            dL_dz_old = dL_dz
            # updating weights
            new_weight = layer._ws - self.lr * dL_dw
            self.layers[-i]._ws = new_weight
            #updating bias
            new_bias = layer._b - self.lr * dL_db
            self.layers[-i]._b = new_bias

    def train(self):
        for epoch in range(self.epochs):
            l = 0
            acc = 0
            self.shuffle()
            
            for batch in range(self.input.shape[0]//self.batch-1):
                start = batch*self.batch
                end = (batch+1)*self.batch
                self.x = self.input[start:end]
                self.y = self.target[start:end]
                self.feedforward()
                self.backprop()
                l+=np.mean(self.error**2)
                acc+= np.count_nonzero(np.argmax(self.layers[-1]._a,axis=1) == np.argmax(self.y,axis=1)) / self.batch
            self.loss.append(l/(self.input.shape[0]//self.batch))
            self.acc.append(acc*100/(self.input.shape[0]//self.batch))
            if epoch % 50 == 0:
                print(f'Model(lr={self.lr},epochs={self.epochs}) : ecpch {epoch} acc:{self.acc[-1]}') 
            
    def plot(self):
        plt.figure(dpi = 125)
        plt.plot(self.loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss vs Epochs")
    
    def acc_plot(self):
        plt.figure(dpi = 125)
        plt.plot(self.acc)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        
    def test(self,xtest,ytest):
        self.x = xtest
        self.y = ytest
        self.feedforward()
        acc = np.count_nonzero(np.argmax(self.layers[-1]._a,axis=1) == np.argmax(self.y,axis=1)) / self.x.shape[0]
        print("Accuracy:", 100 * acc, "%")
        print(np.argmax(self.layers[-1]._a,axis=1))
        return np.argmax(self.layers[-1]._a, axis=1)
