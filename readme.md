# Documentation for AI Path-Following Robot

## Introduction

This document provides an overview of an AI robot built using the Lego Spike Prime platform. The robot consists of two motor-driven wheels with encoders and a supporting ball, enabling free movement. The robot’s key feature is its ability to retake the same path it learned during a teaching phase without memorizing every point in the path. Instead, the robot uses machine learning to generalize the movement and replicate the path effectively.

## System Design

To collect training data, the user simply moves the robot around while keeping a button pressed. During this phase, the robot records:
1. Left wheel velocity over time.
2. Right wheel velocity over time.
3. IMU angle over time.

These data are recorded at a low refresh rate to reduce memory usage while maintaining sufficient granularity for training. The goal is to train a regression model on these graphs to understand the robot’s motion throughout the path rather than memorizing specific points.

## Solutions

Two solutions were developed to achieve the desired behavior. Both share the same architecture for data collection and training but differ in the choice of machine learning algorithm and implementation strategies.

### Solution 1: Neural Network Regression

#### Approach

The first solution used a neural network based on Multi-Layer Perceptrons (MLPs) with ReLU activation functions. This architecture enabled the network to learn complex relationships between time and the recorded data. The network’s structure was initially defined as:



Input: [1] -> [128] -> [128] -> [1]

The neural network took time as an input and produced velocity or IMU angle as the output, effectively running regression on the recorded graphs.

#### Challenges and Observations

The neural network regression faced significant challenges when implemented on the Lego Spike Prime:
- The initial network size was too large for the hardware limitations of the Spike Prime, leading to memory constraints.
- Transitioning the implementation to MicroPython, which lacks libraries like NumPy and PyTorch, resulted in extremely slow computation. Matrix multiplications performed in pure Python took approximately 15 minutes to train a network with just 300 neurons.
- While the neural network captured the overall trends in the data, the continuous outputs were prone to small inaccuracies that compounded during the path-following phase.

To address these issues, the neural network was redesigned to use discrete outputs, significantly reducing computational requirements.

#### Discrete Neural Network Regression

By transitioning to discrete outputs, the neural network became a classifier rather than a regressor. The network categorized the input time into discrete steps and mapped these steps to corresponding velocities or IMU angles. The architecture was simplified to:



Input: [1] -> [20] -> [20]

This approach provided the following advantages:
- Reduced the number of neurons and computational overhead.
- Improved memory efficiency, allowing the network to run on the Spike Prime hardware.
- Enhanced stability during the path-following phase due to discrete outputs.

### Solution 2: Polynomial Regression

#### Approach

The second solution utilized polynomial regression for its simplicity and computational efficiency. Polynomial regression smoothed the recorded data and allowed training directly on the Lego Spike Prime hardware without requiring extensive computational resources.

#### Advantages

- Significantly faster training and inference compared to neural networks.
- Produced smoother and more accurate results during path-following tests.
- Fully compatible with MicroPython, avoiding the need for external libraries.

### Control Mechanism: PID Controller

For both solutions, a PID controller was implemented to maintain the IMU angle during path-following. This ensured improved stability and accuracy compared to relying solely on velocity following.

## Results

Comparison of the outputs of the discrete neural network and polynomial regression showed that while both methods successfully generalized the recorded data, polynomial regression provided a smoother and more efficient solution for the Spike Prime platform.

## Conclusion

This project demonstrates the potential of using machine learning for path-following tasks on resource-constrained platforms like the Lego Spike Prime. Neural networks with discrete outputs offered significant improvements over continuous regression but were limited by hardware constraints. Polynomial regression emerged as the optimal solution due to its simplicity, computational efficiency, and robust performance.

Future work may involve further optimization of training algorithms and exploring hybrid approaches to improve performance while maintaining efficiency.
