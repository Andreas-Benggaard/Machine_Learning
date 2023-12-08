import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

PATH_TO_FASHION_TEST = "../data/fashion_test.npy"
PATH_TO_FASHION_TRAIN = "../data/fashion_train.npy"

def load_data(load_training_data = True):
    """

    function to load fashion data
    
    load_training_data: bool, default: True
        True: loads training data
        False: loads test data

    returns
    ------
    X: list
        list of list of pixel values
    Y: list
        list of list of labels

    """
    if load_training_data:
        data = np.load(PATH_TO_FASHION_TRAIN)
    else:
        data = np.load(PATH_TO_FASHION_TEST)
    X = data[:,:-1] # images
    Y = data[:,-1] # labels : {0,1,2,3,4}
    return X,Y

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
    return normalize(X),one_hot(y)

class Network():
    # TODO: new back probagation
    # TODO: clean up
    # TODO: Softmax
    # TODO: different activations functions
    # TODO: loss funciton
    # TODO: accuracy
    # TODO: identyfy layer?? 
    # TODO: next prev layer??
    # TODO: weight for input/output??
    
    def __init__(self,X, Y):
        # data
        self.X = X
        self.Y = Y
        # size of data
        self._n = self.X.shape[0] 
        self._size_p = self.X.shape[1]
        self._layers = [] # list for the class layers
        # prediction
        self._learning_rate = 0.1
        self._predictions = None
        self._predicted_labels = None
        # 
        self._output_layer = None

    class Layer():
        
        def __init__(self, input_size, output_size):
            # shape
            self._input_size = input_size
            self._output_size =  output_size
            # weights 
            self._ws = self._create_weights(shape=(input_size,output_size), initialization="X")
            # data for layer
            self._x = None
            self._z = None # Xw
            self._b = self._create_bias()
            self._a = None # sigma(Xw + b)
            # activation function
            self._activation_function = None
            # prev and next layers
            self._prev = None
            self._next = None

        def _create_weights(self, shape, initialization = None):
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
        

        def _create_bias(self):
            bias = np.ones((1,self._ws.shape[1]))
            return bias

        def create_bias(shape):
            return np.random.randn(shape[0],shape[1])
        
    
    def add_new_input_layer(self, input_size, output_size):
        layer = self.Layer(input_size, output_size)
        layer._x = self.X
        layer._activation_function = self._relu
        layer._activation_function_der = self._relu_der
        # current layer
        self._layers.append(layer)

    def add_new_hidden_layer(self, input_size, output_size, activation = "relu"):
        layer = self.Layer(input_size, output_size)
        layer._x = None
        if activation == "relu":
            layer._activation_function = self._relu
            layer._activation_function_der = self._relu_der
            # prev layer 
        elif activation == "soft_max":
            layer._activation_function = self._soft_max
        elif activation == "arg_max":
            layer._activation_function = self._arg_max
        self._layers[-1]._next = layer
        # current layer
        layer._prev = self._layers[-1]
        self._layers.append(layer)
        
    def add_new_output_layer(self, input_size, output_size):
        layer = self.Layer(input_size, output_size)
        layer._x = None
        layer._activation_function = self._arg_max
        # prev layer
        self._layers[-1]._next = layer
        # current layer
        layer._prev = self._layers[-1]
        # network
        self._output_layer = layer
 
    def run_forward(self, x):
        """
        pass through the network and calculates as, zs and the predicted labels
        """
        ## TODO: Different activations functions
        next_x = None
        if not isinstance(x, np.ndarray):
            x = self.X
        for i, layer in enumerate(self._layers):# runs trough all the weights
            # if not input layer
            if isinstance(next_x, np.ndarray):
                layer._x = next_x
            # z = Xw + b
            layer._z = layer._x @ layer._ws + layer._b
            # activate result
            layer._a = layer._activation_function(layer._z) # a
            # for next iteration
            next_x = layer._a
        # out put layerf
        self._predicted_labels = next_x    
        return self._predicted_labels
    
    def _back_propagation(self, y_train):
        """
        """
        first = True
        i = 1
        k = len(self._layers) + 1 
        while i < k:
            layer = self._layers[-i]
            if first:
                dz_dw = layer._x # ah
                # print("y:", y_train.shape)
                # ao - y ((n, output neurons) - (n, 1))
                dL_dz = 2*(self._predicted_labels - self._one_hot_Y(y_train))# dL_dzo.shape = (n, output neurons)
                # gradient foor the weight in the output layer
                dL_dw = dz_dw.T @ dL_dz # shape dL_dw = same as weights (in_neurons, out neurons)
                # bias
                dL_db = dL_dz.sum(axis=0).reshape(1,-1)
                first = False
            else: # other than last hidden layer
                # weight 
                dah_dzh = layer._activation_function_der(layer._z)
                # gradient for the weights hidden layer
                dL_dw = layer._x.T @ dah_dzh
                # bias
                dzo_dah = old_weight
                dL_dah = dL_dz_old @ dzo_dah.T # dL_dah.shape == dah_dzh.shape
                # print(dL_dah.shape, dah_dzh.shape)
                dL_dz = dL_dah * dah_dzh 
                dL_db = dL_dz.sum(axis=0).reshape(1,-1) # shape dL_db = bias 
            # weights
            old_weight = layer._ws
            dL_dz_old = dL_dz
            new_weight = layer._ws - self._learning_rate * dL_dw
            self._layers[-i]._ws = new_weight
            # bias
            new_bias = layer._b -  self._learning_rate * dL_db
            self._layers[-i]._b = new_bias
            i += 1

    def train_nework(self, x_train, y_train, learning_rate, epochs):
        # set learning rate
        self._learning_rate = learning_rate
        for i in range(epochs):
            # go throgh network and calculate everythong
            y_hat = self.run_forward(x_train)
            # pass back through the network and update weights
            self._back_propagation(y_train)
            # get predicted labels
            # y_hat = self._get_predicted_labels()
            yy = self._argmax(y_hat)
            acc = self._calculate_accuracy(y_train, yy)
            print("epoch:",i,"ac:",acc)
            # calculate loss
            # loss = self._calulate_loss(y_train, y_hat)
            if i == epochs-1:
                print(yy)
            # print(f"epoch: {i} loss: {loss} accurracy: " )
    
    
    def _soft_max(self, x):
        exp = np.exp(x - np.max(x))
        return exp / exp.sum(axis=1).reshape(-1,1)

    def _calculate_accuracy(self, y_true, y_pred):
        n = y_true.shape[0]
        correct = y_true == y_pred
        accuracy = np.sum(correct) / n
        return accuracy

    def _arg_max(self, x):
        return np.argmax(x, axis = 1)

    def _one_hot_Y(self, y_train):
        shape = (y_train.shape[0],5)
        one_hot_y = np.zeros(shape)
        for i in range(y_train.shape[0]):
            one_hot_y[i][y_train[i]] = 1
        return one_hot_y
    # ----------------------------

    
    
   

    def _print_created_network(self):
        return print(self)
    
    def __str__(self) -> str:
        output = ""
        for i,layer in enumerate(self._layers):
            output += f" {layer}: neurons: {self._neurons_in_layers[i]} \n"
        return output

   

   

    def accuracy(self, y_true, y_hat):
        correct_predictions = y_true == y_hat

    def _calulate_loss(self, y_true, y_pred):
        result = - self.mse(y_true, y_pred)
        return result
    
    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def mse_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_pred.size

        
    def _backwards_output(self):
        pass
    
    def _get_reversed_list(self, x):
        """
        returns the list backwards
        """
        r_list = list(reversed(x))
        return r_list
        
    def _sigmoid(self, x):
        """
        activation function
        """
        result = 1/(np.exp(-x))
        return result

    def _sigmoid_der(self, x): 

        result = self.sigmoid(x) * (1 - self.sigmoid(x))
        return result
    
    def _relu(self, x):
        return np.maximum(x, 0)

    def _relu_der(self, x):
        return np.where(x > 0, 1, 0)


    def predict(self, x):
        # clear prevous predictions
        self._predictions.clear()
        self._predicted_labels.clear()
        # go throug network
        self.pass_forward(x)
        # get predicted labels
        labels = self._get_predicted_labels()
        return labels
        
    def _print_result_for_epoch(self):
        return self._ws[-1]

    def _argmax(self, x):
        result = np.argmax(x, axis=1)
        return result
    def _get_predicted_labels(self):
        return self._predicted_labels[0]
    def get_predicted_labels(self):
        return self._get_predicted_labels()
        
    def get_ws(self):
        return self._ws

    def count_of_layers(self):
        return self._k

    def get_activated(self):
        return self._a

    def get_neurons_in_layers(self):
        return self._neurons_in_layers

    def _add_ones(self, x):
        return np.c_[x, np.ones(x.shape[0])] # add a column of ones     

def main():
    pass

if __name__ == "__main__":
    main()