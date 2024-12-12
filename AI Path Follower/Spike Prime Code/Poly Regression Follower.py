# Project: Polynomial Regression AI Path-Following Robot
# Author: Codrin Crismariu
# Date: November 2024
# Description: This project implements a polynomial regression-based path-following AI 
#              for the Lego Spike Prime robot. The robot records IMU angles and motor
#              velocities during a training phase, fits polynomials to the data, and
#              uses them to follow the recorded path.

# Import necessary modules
from hub import button, light_matrix, motion_sensor, port
import motor
import time

# Define ports for left and right motors
port_left = port.E
port_right = port.F

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

# Fits a polynomial of the given degree to the data using least squares.
# Returns the coefficients of the polynomial.
def fit_polynomial(x, y, degree):
    n = len(x)
    A = [[sum(xi**(i+j) for xi in x) for j in range(degree+1)] for i in range(degree+1)]
    B = [sum(yi * (xi**i) for xi, yi in zip(x, y)) for i in range(degree+1)]
    coefficients = solve_linear_system(A, B)
    return coefficients

# Solve a linear system of equations using Gaussian elimination
def solve_linear_system(A, B):
    n = len(B)
    for i in range(n):
        # Normalize the pivot row
        diag_factor = A[i][i]
        for j in range(i, n):
            A[i][j] /= diag_factor
        B[i] /= diag_factor

        # Eliminate entries below the pivot
        for k in range(i+1, n):
            factor = A[k][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
            B[k] -= factor * B[i]

    # Back substitution
    x = [0] * n
    for i in range(n-1, -1, -1):
        x[i] = B[i] - sum(A[i][j] * x[j] for j in range(i+1, n))
    return x

# Predicts the value of the polynomial with the given coefficients at x.
def predict_polynomial(coefficients, x):
    return sum(c * (x**i) for i, c in enumerate(coefficients))

# Initialize variables for storing processed data and polynomial coefficients
input_data, IMU_data, VEL_A_data, VEL_B_data = [], [], [], []
IMU_coefficients = []
VEL_A_coefficients = []
VEL_B_coefficients = []

# Displays the progress of the polynomial fitting on the robot's light matrix.
def completeMatrix(val):
    light_matrix.clear()
    for i in range(int(val)):
        light_matrix.set_pixel(i % 5, int(i / 5), 100)

# Main loop
while True:
    # Check if the left button is pressed to start recording
    if button.pressed(button.LEFT):
        light_matrix.clear()
        training_data = []

        while button.pressed(button.LEFT):
            # Record training data at regular intervals
            training_data.append(record_data())
            time.sleep(0.1)

        # Process training data and fit polynomials
        input_data, IMU_data, VEL_A_data, VEL_B_data = process_training_data()

        IMU_coefficients = fit_polynomial(input_data, IMU_data, degree=20)
        completeMatrix(25 / 3)  # Display progress
        VEL_A_coefficients = fit_polynomial(input_data, VEL_A_data, degree=20)
        completeMatrix(50 / 3)  # Display progress
        VEL_B_coefficients = fit_polynomial(input_data, VEL_B_data, degree=20)
        completeMatrix(25)  # Display progress

        # Clear processed data after fitting
        input_data, IMU_data, VEL_A_data, VEL_B_data = [], [], [], []

    # Check if the right button is pressed to start path-following
    if button.pressed(button.RIGHT):
        start_time = time.ticks_ms()

        while time.ticks_ms() - start_time < RUN_TIME_MS:
            delta_time = (time.ticks_ms() - start_time) / RUN_TIME_MS

            # Predict desired values from polynomials
            desired_imu = predict_polynomial(IMU_coefficients, delta_time)
            yaw, pitch, roll = motion_sensor.tilt_angles()
            imu_error = wrap_angle(desired_imu - yaw)
            IMU_P = 5  # Proportional gain for IMU control (Don't need the full PID controller for these purposes)

            # Honestly I have no clue why you have to multiply by 10 here but turns out the velocity that you read 
            # from the motor is not in the same scale as the one you set it to. So you have to multiply by 10 ¯\_(ツ)_/¯
            vel_a = predict_polynomial(VEL_A_coefficients, delta_time) * 10
            vel_b = predict_polynomial(VEL_B_coefficients, delta_time) * 10

            # Adjust motor velocities with IMU correction
            motor.run(port_left, int(vel_a + IMU_P * imu_error))
            motor.run(port_right, int(vel_b + IMU_P * imu_error))

        # Stop motors after completing the path
        motor.stop(port_left)
        motor.stop(port_right)
