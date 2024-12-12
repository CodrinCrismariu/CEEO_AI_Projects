import numpy as np
import matplotlib.pyplot as plt
import time

class SimpleNeuralNetwork:
    def __init__(self, input_size=1, hidden_size=10, output_size=10, learning_rate=10, lr_decay=0.999, seed=43):
        # Set parameters
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        
        # Initialize weights and biases
        self.W1 = np.random.normal(0, 0.01, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.normal(0, 0.01, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(sx):
        return sx * (1 - sx)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def forward_pass(self, x_data):
        z1 = np.dot(x_data, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        y_pred = self.softmax(z2)
        return a1, y_pred
    
    def backward_pass(self, x_data, y_true_one_hot, a1, y_pred):
        m = y_true_one_hot.shape[0]
        
        dz2 = y_pred - y_true_one_hot
        dW2 = np.dot(a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(a1)
        dW1 = np.dot(x_data.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return dW1, db1, dW2, db2
    
    def update_parameters(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, x_data, y_data, epochs):
        y_min, y_max = np.min(y_data), np.max(y_data)
        y_true = ((y_data - y_min) / (y_max - y_min) * (self.output_size - 1)).astype(int)

        for epoch in range(epochs):
            self.learning_rate *= self.lr_decay
            a1, y_pred = self.forward_pass(x_data)
            loss = self.cross_entropy_loss(y_true, y_pred)
            
            y_true_one_hot = np.eye(self.output_size)[y_true]
            dW1, db1, dW2, db2 = self.backward_pass(x_data, y_true_one_hot, a1, y_pred)
            self.update_parameters(dW1, db1, dW2, db2)
    
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Learning Rate: {self.learning_rate:.4f}')
    
    def predict(self, x_data):
        _, y_pred = self.forward_pass(x_data)
        return y_pred
    
    def plot_results(self, x_data, y_true, y_pred, poly_fit=None):
        y_max, y_min = np.max(y_true), np.min(y_true)
        y_pred_rescaled = np.argmax(y_pred, axis=1) / (self.output_size - 1) * (y_max - y_min) + y_min

        plt.scatter(x_data, y_true, color='blue', label='True Data')
        #plt.plot(x_data, y_pred_rescaled, color='red', label='NN Predicted Data')
        if poly_fit is not None:
            plt.plot(x_data, poly_fit, color='green', label='Polynomial Fit')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Results Comparison')
        plt.show()

# Polynomial Regression and Neural Network Example
x_data = np.linspace(0, 1, 50).reshape(-1, 1)
y_data = np.array([0, 0, -5, 0, 0, 0, 0, -22, -28, -28, -29, -31, -33, -32, -29, -34, -35, -32, -34, -38,
                   -38, -37, -34, -39, -38, -38, -30, -26, -26, -26, -22, -16, -17, -18, -15, -17, -18,
                   -19, -20, -16, -21, -23, -15, -20, -29, -30, -31, -36, -32, -23])

DISCRETE_CLASSES = 40
DEGREE = 20

# Fit Polynomial Regression
poly_coeffs = np.polyfit(x_data.flatten(), y_data, DEGREE)
poly_fit = np.polyval(poly_coeffs, x_data.flatten())

# Create and train the neural network
nn = SimpleNeuralNetwork(hidden_size=40, output_size=DISCRETE_CLASSES)
start_time = time.time()
nn.train(x_data, y_data, epochs=2000)
print("Training time: ", time.time() - start_time)

# Predict with Neural Network
y_pred = nn.predict(x_data)

# Plot the results
nn.plot_results(x_data, y_data, y_pred, poly_fit=poly_fit)
