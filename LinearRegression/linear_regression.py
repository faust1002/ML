#!/usr/bin/python3

'''Simple implementation of linear regression

   The output is given by a linear relation y = X * theta
   where both X and theta are vectors.
   Number of entries in those vectors depends on number of features (d)
   that are used to estimate the output. Therefore X is 1 x d
   and theta is d x 1. The goal is to estimate theta (function parameters)
   given X and y.

   This approach can be extended futher. More than one pair (X, y)
   can be used to find out theta. Number of samples (pairs) is given by n.
   Therefore X is n x d and theta d x n.

   First, the plain old good least squares method is used to estimate theta.
   Next, theta is found out using gradient descent. Finally stochastic gradient
   descent is applied.'''

import argparse

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

def create_training_data(number_of_samples, number_of_features = 1, mean = 0, variance = 1):
    '''Generate set of example inputs and outputs to be used as a training set.
       The outputs are simply linear function of inputs calculated with a random slope and bias.
       On top of that, a random noise is added to spicy things up a little bit.'''

    # First calculate the set of example inputs - X
    step = np.random.randn(number_of_features, number_of_samples)
    X = np.linspace(0.0, 1, number_of_samples)
    # Each value in the row vector X is multiplied by a corresponding value in step matrix
    # resulting in number_of_features by number_of_features matrix
    X = step * X
    # Because the least square method calculates (X^T * X)^-1, (X^T * X) cannot be singular
    # (X^T * X) will be full rank if and only if the rank of X
    # is equal to min(number_of_features, number_of_samples)
    expected_rank = np.min((number_of_features, number_of_samples))
    actual_rank = np.linalg.matrix_rank(X)
    while actual_rank != expected_rank:
        print("The previously calculated input matrix was singular, recalculating")
        # Recalculate the rank of X does not meet the expectations
        step = np.random.randn(number_of_features, 1)
        X = np.linspace(0.0, 1, number_of_samples)
        X = step * X
        actual_rank = np.linalg.matrix_rank(X)
    X = X.transpose()
    # Next calculate y as a linear function of X
    # Slope and bias are selected randomly for each feature separately
    slope = np.random.randint(1, 5, (number_of_features, 1))
    # Keep the bias small, otherwise the generated data is kind of rubbish
    bias = np.sqrt(0.1) * np.random.randn(number_of_samples, 1)
    y = X.dot(slope) + bias

    # Finally add some Gaussian noise on top of calculated y
    # Because I am lazy, mean and variance are common across all feautures
    noise = np.sqrt(variance) * np.random.randn(number_of_samples, 1) + mean
    y = y + noise
    return (X, y)

def estimate_theta_using_least_squares(X, y):
    '''Estimate theta using least squares method
       theta_hat = (X^T * X)^-1 * X^T * y'''

    theta_hat = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    return theta_hat

def estimate_theta_using_gradient_descent(X, y, number_of_iterations,
                                          learning_rate = 0.01):
    '''Estimate theta using gradient descent'''
    number_of_samples, number_of_features = X.shape
    theta_hat = np.zeros(number_of_features)
    for _ in range(number_of_iterations):
        corrections = np.zeros(number_of_features)
        # Sum over all samples ((theta^T * X[sample] - y[sample]) * X[sample])
        for sample in range(number_of_samples):
            # Theoretically, because theta_hand and X[sample, :] are 1-D array,
            # the theta_hat.traspose() is not needed
            corrections += (theta_hat.transpose().dot(X[sample, :]) - y[sample, :]) * X[sample, :]

        theta_hat -= learning_rate * corrections

    # Return 2-D array
    theta_hat = theta_hat.reshape(number_of_features, 1)
    return theta_hat

def estimate_theta_using_stochastic_gradient_descent(X, y, number_of_iterations,
                                                     learning_rate = 0.01):
    '''Estimate theta using gradient descent'''
    number_of_samples, number_of_features = X.shape
    theta_hat = np.zeros(number_of_features)
    for _ in range(number_of_iterations):
        # Theoretically, because theta_hand and X[sample, :] are 1-D array,
        # the theta_hat.traspose() is not needed
        sample = np.random.randint(0, number_of_samples - 1)
        corrections = (theta_hat.transpose().dot(X[sample, :]) - y[sample, :]) * X[sample, :]
        theta_hat -= learning_rate * corrections

    # Return 2-D array
    theta_hat = theta_hat.reshape(number_of_features, 1)
    return theta_hat

def calculate_mean_squared_error(y, y_hat):
    '''Calculate mean squared error between the training output
       and the output calculated using the estimated paramaters.
       The result is rounded to two decimal places'''

    mse = np.mean(np.square(y - y_hat))
    mse = np.round(mse, decimals = 2)
    return mse

def main():
    '''The main function'''

    # First parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', default = False, type = bool)
    parser.add_argument('-f', '--number_of_features', default = 1, type = int)
    parser.add_argument('-n', '--number_of_samples', default = 100, type = int)
    parser.add_argument('-i', '--number_of_iterations', default = 500, type = int)
    args = parser.parse_args()

    print("Hello")

    # Generate the training set
    print("Step 1 - generate training data")
    X, y = create_training_data(args.number_of_samples, args.number_of_features, variance = 1)

    # Estimate theta using the least squares method
    print("Step 2 - calculate theta using least squares method")
    theta_hat_least_squares = estimate_theta_using_least_squares(X, y)
    y_hat_least_squares = X.dot(theta_hat_least_squares)

    # Estimate theta using the gradient descent method
    print("Step 3 - calculate theta using gradient descent")
    theta_hat_gd = estimate_theta_using_gradient_descent(X, y, args.number_of_iterations)
    y_hat_gd = X.dot(theta_hat_gd)

    # Estimate theta using the stochastic gradient descent method
    print("Step 4 - calculate theta using stochastic gradient descent")
    theta_hat_sgd = estimate_theta_using_stochastic_gradient_descent(X, y,
                                                                    args.number_of_iterations)
    y_hat_sgd = X.dot(theta_hat_sgd)

    # Calculate MSE
    print("Step 5 - calculate MSE")
    mse_least_squares = calculate_mean_squared_error(y, y_hat_least_squares)
    mse_gd = calculate_mean_squared_error(y, y_hat_gd)
    mse_sgd = calculate_mean_squared_error(y, y_hat_sgd)
    print("Least squares method MSE = ", mse_least_squares)
    print("Gradient descent method MSE = ", mse_gd)
    print("Stochastic gradient descent method MSE = ", mse_sgd)

    # Plotting makes sense only if number of features is set to 1
    if args.plot is True and args.number_of_features == 1:
        matplotlib.use('TkAgg')
        # Plot the output value against the values of the first features
        plt.scatter(X[:, 0], y[:, 0], label = 'Training data')
        plt.plot(X[:, 0], y_hat_least_squares[:, 0], 'r', label = 'Least squares estimate')
        plt.plot(X[:, 0], y_hat_gd[:, 0], 'og', label = 'Gradient descent')
        plt.plot(X[:, 0], y_hat_sgd[:, 0], 'xm', label = 'Stochastic gradient descent')
        plt.title('Feature values against output values')
        plt.xlabel('Feature values')
        plt.ylabel('Output values')
        plt.legend(loc='lower right')
        plt.grid(True)

        print("Plot ready")
        plt.show()

    print("Goodbye")

if __name__ == "__main__":
    main()
