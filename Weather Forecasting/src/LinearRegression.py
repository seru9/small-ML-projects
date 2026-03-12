# Credits to https://github.com/mahdi-eth/Linear-Regression-from-Scratch
import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000, penalty = None):
        self.lr = lr # Learning Rate
        self.n_iters = n_iters # max iterations
        self.weights = None
        self.bias = None
        self.penalty = penalty
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        tol = 1e-5
        prev_loss = 0
        for i in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias # linear function
            dw = (1/n_samples) * (np.dot(X.T, (y_pred - y))) # gradient descent for wages
            # add regularization derevitive terme
            if self.penalty == 'l1': # in order to avoid overfitting we are regulating
                dw += self.lr * np.sign(self.weights)
            elif self.penalty == 'l2':
                dw += self.lr * 2 * self.weights
            db = (1/n_samples) * (np.sum(y_pred - y)) # gradient descent for biases
            self.weights = self.weights - (self.lr * dw) # normalization?
            self.bias = self.bias - (self.lr * db)
            
            current_loss = np.mean(np.square(y_pred - y))
            
            if abs(current_loss - prev_loss) < tol:
                break
                
            prev_loss = current_loss
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias