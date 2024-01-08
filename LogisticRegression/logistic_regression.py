#!/usr/bin/python3

'''Simple logistic regression implementation'''

import argparse

import numpy as np

def create_data(number_of_points, shift = 1, mean = 0, variance = 0.1):
    '''Generate training data. Since this data will be used
       to train Perceptron model only two classes are created.'''

    # All data share the same slope and bias. The second class
    # is simply shifted version of the first one
    slope = np.random.randint(1, 10)
    bias = np.random.randint(1, 10)

    # Calculate the first class
    t = np.linspace(0, 1, number_of_points)
    X = slope * t + bias
    y = np.zeros(number_of_points)

    # Calculate the second class
    number_of_points_per_class = int(np.floor(number_of_points / 2))
    # This pushes a limit a little bit since there is no guarantee
    # that np.random.randint will generate unique values
    idx = np.random.randint(0, number_of_points, number_of_points_per_class)
    X[idx] = slope * (t[idx] - shift) + bias
    y[idx] = 1

    # A little bit of noise never hurted anybody
    noise = np.sqrt(variance) * np.random.randn(number_of_points) + mean
    X += noise

    return X, y

def h_theta(theta, x):
    '''The classification function used first to
       find theta_hat and then to calculate
       the new classification'''
    z = theta.transpose().dot(x)
    g = 1 / (1 + np.exp(-z))
    return g

def estimate_theta_hat(X, y, learning_rate, number_of_iterations):
    '''Estimate theta_hat using Perceptron algorithm'''
    number_of_samples, = X.shape
    number_of_features = 1
    theta_hat = np.zeros(number_of_features)
    for _ in range(number_of_iterations):
        corrections = np.zeros(number_of_features)
        for sample in range(number_of_samples):
            corrections += (y[sample] - h_theta(theta_hat, X[sample]))

        theta_hat += learning_rate * corrections

    return theta_hat

def main():
    '''The main function'''

    # First parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number_of_points', default = 100, type = int)
    parser.add_argument('-i', '--number_of_iterations', default = 500, type = int)
    parser.add_argument('-s', '--shift', default = 1, type = int)
    parser.add_argument('-l', '--learning_rate', default = 0.01, type = float)
    args = parser.parse_args()

    print("Hello")

    # Generate the data
    X, y = create_data(args.number_of_points, args.shift)
    # Estimate theta_hat
    theta_hat = estimate_theta_hat(X, y, args.learning_rate, args.number_of_iterations)

    # Reclassify data and find out how many points were marked incorrectly
    y_hat = np.rint([h_theta(theta_hat, x) for x in X]).reshape(args.number_of_points)
    misclassified_points = np.sum(y_hat != y)
    misclassificated_points = (misclassified_points / args.number_of_points) * 100
    misclassificated_points = np.round(misclassificated_points, decimals = 2)
    print(f"Estimated theta_hat = {theta_hat}, misclassifcated points = {misclassificated_points}%")

    print("Goodbye")

if __name__ == "__main__":
    main()
