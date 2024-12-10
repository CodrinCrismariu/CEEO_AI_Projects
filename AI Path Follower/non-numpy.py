import random
import math
import matplotlib.pyplot as plt
import time

class SimpleNeuralNetwork:
    def __init__(self, input_size=1, hidden_size=10, output_size=10, learning_rate=10, lr_decay=0.999, seed=43):
        # Set parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        
        # Initialize weights and biases
        random.seed(seed)
        self.W1 = [[random.gauss(0, 0.01) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [0.0] * hidden_size
        self.W2 = [[random.gauss(0, 0.01) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [0.0] * output_size
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(sx):
        return sx * (1 - sx)
    
    @staticmethod
    def softmax(x):
        max_x = max(x)
        exp_x = [math.exp(i - max_x) for i in x]
        sum_exp_x = sum(exp_x)
        return [i / sum_exp_x for i in exp_x]
    
    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        return sum(-math.log(y_pred[i][y_true[i]]) for i in range(len(y_true))) / len(y_true)
    
    @staticmethod
    def mat_mult(A, B):
        return [[sum(a * b for a, b in zip(row_a, col_b)) for col_b in zip(*B)] for row_a in A]
    
    @staticmethod
    def add_bias(A, b):
        return [[a + bi for a, bi in zip(row, b)] for row in A]
    
    def forward_pass(self, x_data):
        z1 = self.add_bias(self.mat_mult(x_data, self.W1), self.b1)
        a1 = [[self.sigmoid(z) for z in row] for row in z1]
        z2 = self.add_bias(self.mat_mult(a1, self.W2), self.b2)
        y_pred = [self.softmax(row) for row in z2]
        return a1, y_pred
    
    def backward_pass(self, x_data, y_true_one_hot, a1, y_pred):
        m = len(y_true_one_hot)
        dz2 = [[y_p - y_t for y_p, y_t in zip(yp, yt)] for yp, yt in zip(y_pred, y_true_one_hot)]
        dW2 = [[sum(a1[i][h] * dz2[i][o] for i in range(m)) / m for o in range(self.output_size)] for h in range(self.hidden_size)]
        db2 = [sum(dz2[i][o] for i in range(m)) / m for o in range(self.output_size)]
        
        dz1 = [[sum(dz2[i][o] * self.W2[h][o] * self.sigmoid_derivative(a1[i][h]) for o in range(self.output_size)) for h in range(self.hidden_size)] for i in range(m)]
        dW1 = [[sum(x_data[i][0] * dz1[i][h] for i in range(m)) / m for h in range(self.hidden_size)] for j in range(self.input_size)]
        db1 = [sum(dz1[i][h] for i in range(m)) / m for h in range(self.hidden_size)]
        
        return dW1, db1, dW2, db2
    
    def update_parameters(self, dW1, db1, dW2, db2):
        self.W1 = [[w - self.learning_rate * dw for w, dw in zip(row_w, dW)] for row_w, dW in zip(self.W1, dW1)]
        self.b1 = [b - self.learning_rate * db for b, db in zip(self.b1, db1)]
        self.W2 = [[w - self.learning_rate * dw for w, dw in zip(row_w, dW)] for row_w, dW in zip(self.W2, dW2)]
        self.b2 = [b - self.learning_rate * db for b, db in zip(self.b2, db2)]
    
    def train(self, x_data, y_data, epochs):
        y_min, y_max = min(y_data), max(y_data)
        y_true = [(int((y - y_min) / (y_max - y_min) * (DISCRETE_CLASSES - 1))) for y in y_data]

        for epoch in range(epochs):
            self.learning_rate *= self.lr_decay
            a1, y_pred = self.forward_pass(x_data)
            loss = self.cross_entropy_loss(y_true, y_pred)
            
            y_true_one_hot = [[1 if i == y else 0 for i in range(self.output_size)] for y in y_true]
            dW1, db1, dW2, db2 = self.backward_pass(x_data, y_true_one_hot, a1, y_pred)
            self.update_parameters(dW1, db1, dW2, db2)
    
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Learning Rate: {self.learning_rate:.4f}')
    
    def predict(self, x_data):
        _, y_pred = self.forward_pass(x_data)
        return y_pred
    
    def plot_results(self, x_data, y_true, y_pred):
        y_max = max(y_true)
        y_min = min(y_true)
        y_pred_rescaled = [max(range(len(p)), key=lambda i: p[i]) / (self.output_size - 1) * (y_max - y_min) + y_min for p in y_pred]
        print(y_pred_rescaled)
        plt.scatter(x_data, y_true, color='blue', label='True Data')
        plt.plot(x_data, y_pred_rescaled, color='red', label='Predicted Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title('Final Model After Training')
        plt.show()
        

# Usage Example
x_data =[0.0, 0.02040816, 0.04081633, 0.06122449, 0.08163265, 0.1020408, 0.122449, 0.1428571, 0.1632653,    
         0.1836735, 0.2040816, 0.2244898, 0.244898, 0.2653061, 0.2857143, 0.3061224, 0.3265306, 0.3469388,    
         0.367347, 0.3877551, 0.4081633, 0.4285714, 0.4489796, 0.4693878, 0.4897959, 0.5102041, 0.5306122,    
         0.5510204, 0.5714286, 0.5918367, 0.6122449, 0.632653, 0.6530612, 0.6734694, 0.6938776, 0.7142858,    
         0.7346939, 0.7551021, 0.7755102, 0.7959184, 0.8163265, 0.8367347, 0.8571428, 0.877551, 0.8979591,    
         0.9183673, 0.9387755, 0.9591837, 0.9795918, 1.0]
y_data =[0, 0, -5, 0, 0, 0, 0, -22, -28, -28, -29, -31, -33, -32, -29, -34, -35, -32, -34, -38, -38,     
         -37, -34, -39, -38, -38, -30, -26, -26, -26, -22, -16, -17, -18, -15, -17, -18, -19, -20, -16,    
         -21, -23, -15, -20, -29, -30, -31, -36, -32, -23]

DISCRETE_CLASSES = 20

# Reshape x_data for compatibility
x_data = [[x] for x in x_data]

# Create and train the neural network
nn = SimpleNeuralNetwork(hidden_size=20, output_size=DISCRETE_CLASSES)
start_time = time.time()

nn.train(x_data, y_data, epochs=1000)

print("Training time: ", time.time() - start_time)

# Predict and plot the results
y_pred = nn.predict(x_data)
nn.plot_results(x_data, y_data, y_pred)