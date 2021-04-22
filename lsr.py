import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys, new_xs, new_ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
        line: type of function
        equation: parameters of function
    Returns:
        None
    """

    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    
    for i in range(num_segments):
        plt.plot(new_xs[i], new_ys[i])
       
    plt.show()
    

    

        


def linear_line(xs, ys):
    """ 
    Use least squares method to determine parameters of linear line
    """
    # extend matrix with column of 1's
    matrix = np.column_stack((np.ones(xs.shape), xs))

    # Calculation of (𝑋^𝑇*𝑋)^−1*𝑋^𝑇*𝑌
    # First column of fit is the y-intercept and gradient is second column
    return np.linalg.inv(matrix.T.dot(matrix)).dot(matrix.T).dot(ys)

def poly_line(xs, ys):
    """ 
    Use least squares method to predict parameters of polynomial line
    """
    matrix = np.column_stack((np.ones(xs.shape),  xs, np.square(xs), np.power(xs, 3)))
    return np.linalg.inv(matrix.T.dot(matrix)).dot(matrix.T).dot(ys)


def sine_line(xs, ys):
    """ 
    Use least squares method to predict parameters of sinusoidal line
    """
    matrix = np.column_stack((np.ones(xs.shape), np.sin(xs)))
    return np.linalg.inv(matrix.T.dot(matrix)).dot(matrix.T).dot(ys)

def square_error(y, y_hat):
    """
    Calculate error in predicted line using sum squared error i.e. ∑𝑖(𝑦̂𝑖−𝑦𝑖)2  where 𝑦̂𝑖=𝑎+𝑏𝑥𝑖
    """
    return np.sum((y - y_hat) ** 2)

def linear_cve(train_xs, train_ys, test_xs, test_ys):
    weights = linear_line(train_xs, train_ys)
    test_ys_hat = np.column_stack((np.ones(test_xs.shape), test_xs)).dot(weights)
    return square_error(test_ys, test_ys_hat).mean()

def poly_cve(train_xs, train_ys, test_xs, test_ys):
    weights = poly_line(train_xs, train_ys)
    test_ys_hat = np.column_stack((np.ones(test_xs.shape), test_xs, np.square(test_xs), np.power(test_xs, 3))).dot(weights)
    return square_error(test_ys, test_ys_hat).mean()

def sine_cve(train_xs, train_ys, test_xs, test_ys):
    weights = sine_line(train_xs, train_ys)
    test_ys_hat = np.column_stack((np.ones(test_xs.shape), np.sin(test_xs))).dot(weights)
    return square_error(test_ys, test_ys_hat).mean()


def k_fold_cross_val(xs, ys, k):
    """
    1. Shuffle the dataset randomly.
    2. Split the dataset into k groups
    3. For each unique group:
        Take the group as a hold out or test data set
        Take the remaining groups as a training data set
        Fit a model on the training set and evaluate it on the test set
        Retain the evaluation score and discard the model
    Summarize the skill of the model using the sample of model evaluation scores
    """
    # shuffle indices
    indices = np.arange(xs.shape[0])
    np.random.shuffle(indices)
    # Split the dataset into k groups
    split_xs, split_ys = np.array_split(xs[indices], k), np.array_split(ys[indices], k)
    cve_linear = cve_poly = cve_sine = 0
    for i in range(k):

        test_xs, test_ys = split_xs[i], split_ys[i]
        train_xs = np.append(split_xs[:i], split_xs[i+1:])
        train_ys = np.append(split_ys[:i], split_ys[i+1:])
        cve_linear += linear_cve(train_xs, train_ys, test_xs, test_ys)
        cve_poly += poly_cve(train_xs, train_ys, test_xs, test_ys)
        cve_sine += sine_cve(train_xs, train_ys, test_xs, test_ys)
    # print(cve_linear/k)
    # print(cve_poly/k)
    # print(cve_sine/k)
    return cve_linear/k, cve_poly/k, cve_sine/k

   


if __name__ == '__main__':

    points = load_points_from_file(sys.argv[1])

    # split into line segments of 20 points
    x_segments = np.split(points[0], len(points[0]) / 20)
    y_segments = np.split(points[1], len(points[1]) / 20)
    num_segments = len(x_segments)
    total_error = 0

    new_xs_list = []
    new_ys_list =[]

    for i in range(num_segments):
        new_xs = np.linspace(min(x_segments[i]), max(x_segments[i]), 100)
        cve_linear, cve_poly, cve_sine = k_fold_cross_val(x_segments[i], y_segments[i], 10)
        min_cve = min(cve_linear, cve_poly, cve_sine)

        if min_cve == cve_linear:
            weights = linear_line(x_segments[i], y_segments[i])
            ys_hat = test_ys_hat = np.column_stack((np.ones(x_segments[i].shape), x_segments[i])).dot(weights)
            error = square_error(y_segments[i], ys_hat)
            new_ys = np.column_stack((np.ones(new_xs.shape), new_xs)).dot(weights)    


        elif min_cve == cve_poly:
            weights = poly_line(x_segments[i], y_segments[i])
            ys_hat = np.column_stack((np.ones(x_segments[i].shape), x_segments[i], np.square(x_segments[i]), np.power(x_segments[i], 3))).dot(weights)
            error = square_error(y_segments[i], ys_hat)
            new_ys = np.column_stack((np.ones(new_xs.shape), new_xs, np.square(new_xs), np.power(new_xs, 3))).dot(weights)


        elif min_cve == cve_sine:
            weights = sine_line(x_segments[i], y_segments[i])
            ys_hat = np.column_stack((np.ones(x_segments[i].shape), np.sin(x_segments[i]))).dot(weights)
            error = square_error(y_segments[i], ys_hat)
            new_ys = np.column_stack((np.ones(new_xs.shape), np.sin(new_xs))).dot(weights)

        new_xs_list.append(new_xs)
        new_ys_list.append(new_ys)
        total_error += error

    print(total_error)



    # if --plot parameter given, visualise result
    if len(sys.argv) == 3 and sys.argv[2] == "--plot":
        view_data_segments(points[0], points[1], new_xs_list, new_ys_list)
