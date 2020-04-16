# import GradientDescent
import numpy as np

class ScratchLinearRegressor:
    def __init__(self):
        self._coef = []
        self.bias = np.random.rand(1)

    def fit_line(self, x, y, learning_rate = 0.001):
        n_samples = len(x)
        n_features = len(x[0])
        self._coef = np.random.rand(n_features, 1) * 0.05
        for n in range(n_samples):
            # Predictions (np.dot is element wise multiplication)
            # y_hat is (1,1) dimensions
            y_hat =  np.dot(x[n], self._coef) + self.bias

            # Error calculation (Squared error)
            # Error  shape (1,1)
            error = (y_hat-y[n])**2

            # calculating  deriatives
            # Coeficient and x[n] shape is (number of features, 1) 
            self._coef -= (learning_rate * 2 * error * x[n].reshape(-1, 1))
            self.bias -= learning_rate * 2 * error 

    def predict(self, input):
        if not isinstance(input, type(np.array)):
            try:
                input = np.array(input).reshape(1, -1)
            except:
                print(f'Could not convert to numpy array. Input type: {type(input)}')
        # assert(self._coef.shape == input.shape)
        return np.dot(input, self._coef) + self.bias
