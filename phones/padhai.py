import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
import operator
import json
np.random.seed(0)

def shuffle_data(X,Y):
    '''Take X-> data, Y-> Class, returned the shuffled tupple of them
        -- Steps.
        - Take out shape of dataset, increase an column in that shape.
        - Create an zero numpy array using that shape, copy data set to it and let last coloumn empty.
        - Copy ground truth in that last column.
        - Now shuffle the array rowwise.
        - Next task seperate out and return them.
    '''
    shp = np.shape(X)
    new_shp = np.array(shp) + np.array([0,1])
    zero_ar = np.zeros(new_shp)
    zero_ar[:,:-1] = X
    zero_ar[:,-1] = Y
    np.random.shuffle(zero_ar)
    X_train_sfl = zero_ar[:,:-1]
    Y_train_sfl = zero_ar[:,-1]
    return (X_train_sfl,Y_train_sfl)


class MPNeuron:
    
    def __init__(self):
        self.theta = None
        
    def mp_neuron(self, x):
        if sum(x) >= self.theta:
            return 1
        return 0
    
    def fit_brute_force(self, X, Y):
        accuracy = {}
        for theta in range(0, X.shape[1]+1):
            self.theta = theta
            Y_pred = self.predict(X)
            accuracy[theta] = accuracy_score(Y, Y_pred)  
            
        sorted_accuracy = sorted(accuracy.items(), key=operator.itemgetter(1), reverse=True)
        best_theta, best_accuracy = sorted_accuracy[0]
        self.theta = best_theta
        print('theta is {}'.format(best_theta))
        
    def fit(self, X, Y, epochs=10, log=False, display_plot=False, shuffle=True):
        self.theta = (X.shape[1]+1)//2
        if log or display_plot:
            accuracy = {}
        for i in range(epochs):
            if shuffle == True:
                (X,Y) = shuffle_data(X,Y)
            Y_pred = self.predict(X)
            tn, fp, fn, tp = confusion_matrix(Y, Y_pred).ravel()
            if fp > fn and self.theta <= X.shape[1]:
                self.theta += 1
            elif fp < fn and self.theta >= 1:
                self.theta -= 1
            else:
                break
                
            if log or display_plot:
                Y_pred = self.predict(X)
                accuracy[i] = accuracy_score(Y, Y_pred)
        if log:
            with open('mp_neuron_accuracy.json', 'w') as fp:
                json.dump(accuracy, fp)
        if display_plot:
            epochs_, accuracy_ = zip(*accuracy.items())
            plt.plot(epochs_, accuracy_)
            plt.xlabel("Epochs")
            plt.ylabel("Train Accuracy")
            plt.show()
        print('theta is {}'.format(self.theta))
    
    def predict(self, X):
        Y = []
        for x in X:
            result = self.mp_neuron(x)
            Y.append(result)
        return np.array(Y)

class Perceptron:
    
    def __init__(self):
        self.w = None
        self.b = None
        
    def perceptron(self, x):
        return np.sum(self.w * x) + self.b
    
    def fit_brute_force(self, X, Y):
        accuracy = {}
        param_grid = {'w'+str(i):np.linspace(-1, 1, num=1) for i in range(X.shape[1])}
        param_grid['b'] = np.linspace(-1, 1, num=10)

        for parameter in ParameterGrid(param_grid):
            w = []
            for key in ['w'+str(i) for i in range(X.shape[1])]:
                w.append(parameter[key])
                
            self.w = np.array(w)
            self.b = parameter['b']            
            Y_pred = self.predict(X)

            index = tuple(w) + (self.b,)
            accuracy[index] = accuracy_score(Y, Y_pred)
            
        sorted_accuracy = sorted(accuracy.items(), key=operator.itemgetter(1), reverse=True)
        best_parameter, best_accuracy = sorted_accuracy[0]
        
        self.w = np.array(best_parameter[:-1])
        self.b = best_parameter[-1]

    
    def fit(self, X, Y, epochs=10, learning_rate=0.01, log=False, display_plot=False, shuffle = False):
        # initialise the weights and bias
        np.random.seed(3)
        self.w = np.random.randn(1, X.shape[1])*0.01
        self.b = 0.01

        if log or display_plot: 
            accuracy = {}
        for i in range(epochs):
            
            if shuffle == True:
                (X,Y) = shuffle_data(X,Y)
            for x, y in zip(X, Y):
                result = self.perceptron(x)
                if y == 1 and result < 0:
                    self.w += learning_rate*x
                    self.b += learning_rate
                elif y == 0 and result >= 0:
                    self.w -= learning_rate*x
                    self.b -= learning_rate

            if log or display_plot:
                Y_pred = self.predict(X)
                accuracy[i] = accuracy_score(Y, Y_pred)
        if log:
            with open('perceptron_accuracy.json', 'w') as fp:
                json.dump(accuracy, fp)
        if display_plot:
            epochs_, accuracy_ = zip(*accuracy.items())
            plt.plot(epochs_, accuracy_)
            plt.xlabel("Epochs")
            plt.ylabel("Train Accuracy")
            plt.show()
        
    
    def predict(self, X):
        Y = []
        for x in X:
            result = self.perceptron(x)
            Y.append(int(result>=0))
        return np.array(Y)


class PerceptronWithSigmoid:
    
    def __init__(self):
        self.w = None
        self.b = None
        
    def perceptron(self, x):
        return np.sum(self.w * x) + self.b
    
    def sigmoid(self, z):
        return 1. / (1. + np.exp(-z))
    
    def cross_entropy(self,  targets, predictions):
        
        N = predictions.shape[0]
        ce = -np.sum(targets *np.log(predictions))/N
        return ce
    
    def grad_w_mse(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * y_pred * (1 - y_pred) * x
    
    def grad_b_mse(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * y_pred * (1 - y_pred)
    
    def grad_w_ce(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
	#print(y_pred)
        if y == 0:
            return x * y_pred
        if y == 1:
            return -1*x*(1 - y_pred)
        
    def grad_b_ce(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        if y == 0:
            return y_pred
        if y == 1:
            return -1*(1-y_pred)        
    
    
    def fit_brute_force(self, X, Y, scaled_threshold=0.5):
        accuracy = {}
        param_grid = {'w'+str(i):np.linspace(-1, 1, num=1) for i in range(X.shape[1])}
        param_grid['b'] = np.linspace(-1, 1, num=10)
        SCALED_THRESHOLD = scaled_threshold

        for parameter in ParameterGrid(param_grid):
            w = []
            for key in ['w'+str(i) for i in range(X.shape[1])]:
                w.append(parameter[key])
                
            self.w = np.array(w)
            self.b = parameter['b']            
            Y_pred = self.predict(X)
            
            Y_binarized = (Y >= SCALED_THRESHOLD).astype(np.int)
            Y_pred_binarized = (Y_pred >= SCALED_THRESHOLD).astype(np.int)

            index = tuple(w) + (self.b,)
            accuracy[index] = accuracy_score(Y_binarized, Y_pred_binarized)
            
        sorted_accuracy = sorted(accuracy.items(), key=operator.itemgetter(1), reverse=True)
        best_parameter, best_accuracy = sorted_accuracy[0]
        
        self.w = np.array(best_parameter[:-1])
        self.b = best_parameter[-1]
#         print(self.w)
    
    def fit(self, X, Y, epochs=10, learning_rate=0.01, log=False, display_plot=False,shuffle = False, loss = "mse", scaled_threshold=0.5):
        # initialise the weights and bias
        np.random.seed(3)
        self.w = np.random.randn(1, X.shape[1])*0.01
        self.b = 0.01
        if log or display_plot: 
            accuracy = {}
            mse = {}
        for i in range(epochs):
            if shuffle == True:
                (X,Y) = shuffle_data(X,Y)
            dw, db = 0, 0
            for x, y in zip(X, Y):
                if loss == "mse":
                    dw += self.grad_w_mse(x, y)
                    db += self.grad_b_mse(x, y)
                if loss == "cross_entropy":
                    dw += self.grad_w_ce(x, y)
                    db += self.grad_b_ce(x, y)                    
                    
            dw /= len(Y)
            db /= len(Y)
            self.w -= learning_rate*dw
            self.b -= learning_rate*db
            
            SCALED_THRESHOLD = scaled_threshold
            if log or display_plot:
                Y_pred = self.predict(X)
                Y_binarized = (Y >= SCALED_THRESHOLD).astype(np.int)
                Y_pred_binarized = (Y_pred >= SCALED_THRESHOLD).astype(np.int)
                accuracy[i] = accuracy_score(Y_binarized, Y_pred_binarized)
                if loss == "mse":
                    mse[i] = mean_squared_error(Y, Y_pred)
                elif loss == "cross_entropy":
                    mse[i] = self.cross_entropy(Y, Y_pred)
        
        print("loss {}".format(loss))
        if log:
            with open('perceptron_with_sigmoid_accuracy.json', 'w') as fp:
                json.dump(accuracy, fp)
            with open('perceptron_with_sigmoid_mse.json', 'w') as fp:
                json.dump(mse, fp)
        if display_plot:
            epochs_, mse_ = zip(*mse.items())
            plt.plot(epochs_, mse_)
            plt.xlabel("Epochs")
            plt.ylabel("Train Error")
            plt.show()
            epochs_, accuracy_ = zip(*accuracy.items())
            plt.plot(epochs_, accuracy_)
            plt.xlabel("Epochs")
            plt.ylabel("Train Accuracy")
            plt.show()
            
    
    def predict(self, X):
        Y = []
        for x in X:
            result = self.sigmoid(self.perceptron(x))
            Y.append(result)
        return np.array(Y)


