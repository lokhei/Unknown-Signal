import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as lines


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys, line_y):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """

    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    # plt.scatter(xs, ys, c=colour)
    # plt.show()
    

    # fig, ax = plt.subplots()
    # ax.scatter(xs, ys, c=colour)
    # plt.show()
    list_x = np.split(xs, len_data / 20)
    for i in range(0, num_segments):
        line = lines.Line2D(list_x[i], line_y[:, i], c='r')
        ax.add_line(line)
    plt.show()





    # xs = np.split(xs, len_data / 20)
    # for i in range(num_segments):
    #     x_min = min(xs[i])
    #     x_max = max(xs[i])
    #     y_min = 

        


def linear_line(xs, ys):
    # extend matrix with column of 1's
    matrix = np.column_stack((np.ones(xs.shape), xs))

    # Calculation of 𝐴=(𝑋^𝑇*𝑋)^−1*𝑋^𝑇*𝑌
    fit = np.linalg.inv(matrix.T.dot(matrix)).dot(matrix.T).dot(ys)
    # First column of fit is the y-intercept and gradient is second column
    bestfit = fit[0] + fit[1] * xs

    return bestfit

def poly_line(xs, ys):
    matrix = np.column_stack((np.ones(xs.shape),  xs, np.square(xs), np.power(xs, 3)))
    cube_fit = np.linalg.inv(matrix.T.dot(matrix)).dot(matrix.T).dot(ys)
    bestfit = cube_fit[0] + cube_fit[1] * xs + cube_fit[2] * np.square(xs) + cube_fit[3] * np.power(xs, 3)

    return bestfit


def sine_line(xs, ys):
    matrix = np.column_stack((np.ones(xs.shape), np.sin(xs)))
    sine_fit = np.linalg.inv(matrix.T.dot(matrix)).dot(matrix.T).dot(ys)
    bestfit = sine_fit[0] + sine_fit[1] * np.sin(xs)

    return bestfit

def square_error(y, y_hat):
    # Calculate least squares error using ∑𝑖(𝑦̂𝑖−𝑦𝑖)2  where 𝑦̂𝑖=𝑎+𝑏𝑥𝑖
    return np.sum((y - y_hat) ** 2)




if __name__ == '__main__':

    points = load_points_from_file(sys.argv[1])

    # split into line segments of 20 points
    x_segments = np.split(points[0], len(points[0]) / 20)
    y_segments = np.split(points[1], len(points[1]) / 20)
    num_segments = len(x_segments)
    total_error = 0
    


    for i in range(num_segments):
        linear_fit = linear_line(x_segments[i], y_segments[i])
        linear_error = square_error(y_segments[i], linear_fit)
        print(linear_error)

        poly_fit = poly_line(x_segments[i], y_segments[i])
        poly_error = square_error(y_segments[i], poly_fit)
        print(poly_error)

        sine_fit = sine_line(x_segments[i], y_segments[i])
        sine_error = square_error(y_segments[i], sine_fit)
        print(sine_error)

        error = min(linear_error, poly_error, sine_error)
        if error == linear_error:
            print("Linear")
        elif error == poly_error:
            print("Cubic")
        elif error == sine_error:
            print("Sine")
        total_error += error
        
    print(total_error)




    # if --plot parameter given, visualise result
    if len(sys.argv) == 3 and sys.argv[2] == "--plot":
        view_data_segments(points[0], points[1])
