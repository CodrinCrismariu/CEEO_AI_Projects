# Project: Neural Network AI Path-Following Robot
# Author: Codrin Crismariu
# Date: November 2024
# Description: This project implements a neural network regression-based path-following AI 
#              for the Lego Spike Prime robot. The robot records IMU angles and motor
#              velocities during a training phase, fits a neural network to the data, and
#              uses them to follow the recorded path.

# Import necessary modules
from hub import button, motion_sensor, port, light_matrix
import motor
import time
import random
import math


# Captures the current time, IMU tilt angles, and motor velocities.
# Returns a list of [timestamp, yaw angle, left motor velocity, right motor velocity].
def record_data():
    yaw, pitch, roll = motion_sensor.tilt_angles()
    vel_a = motor.velocity(port_left)
    vel_b = motor.velocity(port_right)
    return [time.ticks_ms(), yaw, vel_a, vel_b]

# Time step for recording data and storage variables
time_step = 0.1  # in seconds
training_data = []
RUN_TIME_MS = 0  # Total runtime for the recorded path

# Function to wrap angles to handle cyclic behavior
def wrap_angle(angle):
    while angle > 1800:
        angle -= 3600
    while angle < -1800:
        angle += 3600
    return angle

# Function to normalize and process the recorded training data
def process_training_data():
    """
    Processes recorded data by normalizing time and correcting IMU angle wrapping.
    Returns lists of normalized input times, IMU angles, and motor velocities.
    """
    global training_data, RUN_TIME_MS

    # Normalize time to the range [0, 1]
    start_time = training_data[0][0]
    RUN_TIME_MS = training_data[-1][0] - start_time

    for data in training_data:
        data[0] = (data[0] - start_time) / RUN_TIME_MS

    training_input = []
    IMU_output = []
    VEL_A_output = []
    VEL_B_output = []

    prev_angle = 0
    for data in training_data:
        training_input.append(data[0])

        # Adjust IMU angles to maintain continuity
        # This is done because we can't run regression if the graph jumps from 180 to -180 degrees
        while data[1] < prev_angle and prev_angle - data[1] > 1800:
            data[1] += 3600
        while data[1] > prev_angle and prev_angle - data[1] < -1800:
            data[1] -= 3600

        IMU_output.append(data[1])
        prev_angle = data[1]

        VEL_A_output.append(data[2])
        VEL_B_output.append(data[3])

    return [training_input, IMU_output, VEL_A_output, VEL_B_output]

# Function to collect training data when the left button is pressed
def collect_data():
    global training_data

    while True:
        # Wait for the left button to be pressed
        if button.pressed(button.LEFT):
            training_data = []

            # Keep recording data while the button is held
            while button.pressed(button.LEFT):
                # Record IMU and motor data
                training_data.append(record_data())
                # Regulate loop time to prevent memory overflow
                time.sleep(0.1)

            # Process and return the recorded data
            return process_training_data()

# Neural Network class for predicting motor velocities
class SimpleNeuralNetwork:
    def __init__(self, input_size=1, hidden_size=10, output_size=10, learning_rate=10, lr_decay=0.999, seed=43):
        # Initialize parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

        # Initialize weights and biases with small random values
        random.seed(seed)
        self.W1 = [[random.uniform(0, 0.01) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [0.0] * hidden_size
        self.W2 = [[random.uniform(0, 0.01) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [0.0] * output_size

    @staticmethod
    def sigmoid(x):
        # Sigmoid activation function
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(sx):
        # Derivative of the sigmoid function
        return sx * (1 - sx)

    @staticmethod
    def softmax(x):
        # Softmax function for output layer
        max_x = max(x)
        exp_x = [math.exp(i - max_x) for i in x]
        sum_exp_x = sum(exp_x)
        return [i / sum_exp_x for i in exp_x]

    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        # Calculate cross-entropy loss
        return sum(-math.log(y_pred[i][y_true[i]]) for i in range(len(y_true))) / len(y_true)

    @staticmethod
    def mat_mult(A, B):
        # Matrix multiplication
        return [[sum(a * b for a, b in zip(row_a, col_b)) for col_b in zip(*B)] for row_a in A]

    @staticmethod
    def add_bias(A, b):
        # Add bias to the layer
        return [[a + bi for a, bi in zip(row, b)] for row in A]

    def forward_pass(self, x_data):
        # Forward pass: input -> hidden layer -> output layer
        z1 = self.add_bias(self.mat_mult(x_data, self.W1), self.b1)
        a1 = [[self.sigmoid(z) for z in row] for row in z1]
        z2 = self.add_bias(self.mat_mult(a1, self.W2), self.b2)
        y_pred = [self.softmax(row) for row in z2]
        return a1, y_pred

    def backward_pass(self, x_data, y_true_one_hot, a1, y_pred):
        # Backpropagation to calculate gradients
        m = len(y_true_one_hot)
        dz2 = [[y_p - y_t for y_p, y_t in zip(yp, yt)] for yp, yt in zip(y_pred, y_true_one_hot)]
        dW2 = [[sum(a1[i][h] * dz2[i][o] for i in range(m)) / m for o in range(self.output_size)] for h in range(self.hidden_size)]
        db2 = [sum(dz2[i][o] for i in range(m)) / m for o in range(self.output_size)]

        dz1 = [[sum(dz2[i][o] * self.W2[h][o] * self.sigmoid_derivative(a1[i][h]) for o in range(self.output_size)) for h in range(self.hidden_size)] for i in range(m)]
        dW1 = [[sum(x_data[i][0] * dz1[i][h] for i in range(m)) / m for h in range(self.hidden_size)] for j in range(self.input_size)]
        db1 = [sum(dz1[i][h] for i in range(m)) / m for h in range(self.hidden_size)]

        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2):
        # Update weights and biases using calculated gradients
        self.W1 = [[w - self.learning_rate * dw for w, dw in zip(row_w, dW)] for row_w, dW in zip(self.W1, dW1)]
        self.b1 = [b - self.learning_rate * db for b, db in zip(self.b1, db1)]
        self.W2 = [[w - self.learning_rate * dw for w, dw in zip(row_w, dW)] for row_w, dW in zip(self.W2, dW2)]
        self.b2 = [b - self.learning_rate * db for b, db in zip(self.b2, db2)]

    def train(self, x_data, y_data, epochs):
        # Train the neural network using the provided data
        y_min, y_max = min(y_data), max(y_data)
        y_true = [(int((y - y_min) / (y_max - y_min) * (DISCRETE_CLASSES - 1))) for y in y_data]

        for epoch in range(epochs):
            self.learning_rate *= self.lr_decay
            a1, y_pred = self.forward_pass(x_data)
            loss = self.cross_entropy_loss(y_true, y_pred)

            y_true_one_hot = [[1 if i == y else 0 for i in range(self.output_size)] for y in y_true]
            dW1, db1, dW2, db2 = self.backward_pass(x_data, y_true_one_hot, a1, y_pred)
            self.update_parameters(dW1, db1, dW2, db2)

            print('Epoch: ', epoch+1, '/', epochs, 'Loss:', loss)

    def predict(self, x_data):
        # Predict outputs for given input data
        _, y_pred = self.forward_pass(x_data)
        return y_pred

# Number of discrete velocities the neural network can choose from
DISCRETE_CLASSES = 20

# Collect training data
input_data, IMU_data, VEL_A_data, VEL_B_data = collect_data()

# Reshape input data for compatibility with the neural network
input_data = [[inp] for inp in input_data]

# Train neural networks for left and right motor velocities
# FIY this takes a long time to train that's why we don't train IMU for this one
VEL_A_net = SimpleNeuralNetwork(hidden_size=5, output_size=DISCRETE_CLASSES)
VEL_A_net.train(input_data, VEL_A_data, epochs=300)

VEL_B_net = SimpleNeuralNetwork(hidden_size=5, output_size=DISCRETE_CLASSES)
VEL_B_net.train(input_data, VEL_B_data, epochs=300)

# Replay the learned path when the right button is pressed
while True:
    if button.pressed(button.RIGHT):
        start_time = time.ticks_ms()

        # Replay the path for the recorded runtime
        while time.ticks_ms() - start_time < RUN_TIME_MS:
            delta_time = (time.ticks_ms() - start_time) / RUN_TIME_MS

            vel_a = VEL_A_net.predict(x_data=[[delta_time]])
            vel_a_max = max(VEL_A_data)
            vel_a_min = min(VEL_A_data)
            vel_a_rescaled = [max(range(len(p)), key=lambda i: p[i]) / (DISCRETE_CLASSES - 1) * (vel_a_max - vel_a_min) + vel_a_min for p in vel_a]
            vel_b = VEL_B_net.predict(x_data=[[delta_time]])
            vel_b_max = max(VEL_B_data)
            vel_b_min = min(VEL_B_data)
            vel_b_rescaled = [max(range(len(p)), key=lambda i: p[i]) / (DISCRETE_CLASSES - 1) * (vel_b_max - vel_b_min) + vel_b_min for p in vel_b]

            # Apply predicted velocities to the motors
            # Again no clue why we have to multiply by 10 the scales are just weird for the LEGO motors
            motor.run(port.A, int(vel_a_rescaled[0] * 10))
            motor.run(port.B, int(vel_b_rescaled[0] * 10))

        # Stop the motors after completing the path
        motor.run(port.A, 0)
        motor.run(port.B, 0)
