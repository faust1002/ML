#!/usr/bin/python3

'''Simple Perceptron implementation'''

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
    if z >= 1:
        return 1

    return 0

def estimate_theta_hat(X, y, learning_rate):
    '''Estimate theta_hat using Perceptron algorithm'''
    number_of_samples, = X.shape
    number_of_features = 1
    theta_hat = np.zeros(number_of_features)
    for sample in range(number_of_samples):
        theta_hat = theta_hat + learning_rate * (y[sample] - h_theta(theta_hat, X[sample]))

    return theta_hat

def main():
    '''The main function'''

    # First parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number_of_points', default = 100, type = int)
    parser.add_argument('-s', '--shift', default = 1, type = int)
    parser.add_argument('-l', '--learning_rate', default = 0.5, type = float)
    args = parser.parse_args()

    print("Hello")

    # Generate the data
    X, y = create_data(args.number_of_points, args.shift)
    # Estimate theta_hat
    theta_hat = estimate_theta_hat(X, y, args.learning_rate)

    # Reclassify data and find out how many points were marked incorrectly
    y_hat = np.array([h_theta(theta_hat, x) for x in X])
    misclassified_points = np.sum(y_hat != y)
    misclassificated_points = (misclassified_points / args.number_of_points) * 100
    misclassificated_points = np.round(misclassificated_points, decimals = 2)
    print(f"Estimated theta_hat = {theta_hat}, misclassifcated points = {misclassificated_points}%")

    print("Goodbye")

if __name__ == "__main__":
    main()
