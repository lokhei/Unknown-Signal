import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

linear = 1
polynomial = 2
sine = 3

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys, line, equation):
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


    x_segments = np.split(points[0], len(points[0]) / 20)
    y_segments = np.split(points[1], len(points[1]) / 20)


    for i in range(num_segments):
        x_min = min(x_segments[i])
        x_max = max(x_segments[i])
        eq = equation[i]
        if line[i] == linear:
            y_min = eq[0] + eq[1] * x_min
            y_max = eq[0] + eq[1] * x_max
            plt.plot([x_min, x_max], [y_min, y_max])
        elif line[i] == polynomial:
            new_xs = np.linspace(x_min, x_max, 100)
            new_ys_hat = eq[0] + eq[1] * new_xs + eq[2] * np.square(new_xs) + eq[3] * np.power(new_xs, 3)
            plt.plot(new_xs, new_ys_hat)
        elif line[i] == sine:
            new_xs = np.linspace(x_min, x_max, 100)
            new_ys_hat = eq[0] + eq[1] * np.sin(new_xs) 
            plt.plot(new_xs, new_ys_hat)

       
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




if __name__ == '__main__':

    points = load_points_from_file(sys.argv[1])

    # split into line segments of 20 points
    x_segments = np.split(points[0], len(points[0]) / 20)
    y_segments = np.split(points[1], len(points[1]) / 20)
    num_segments = len(x_segments)
    total_error = 0
    

    line = [0] * num_segments
    equation = [0] * num_segments

    for i in range(num_segments):
        linear_fit = linear_line(x_segments[i], y_segments[i])
        linear_ys = linear_fit[0] + linear_fit[1] * x_segments[i]
        linear_error = square_error(y_segments[i], linear_ys)
        # print(linear_error)

        poly_fit = poly_line(x_segments[i], y_segments[i])
        poly_ys = poly_fit[0] + poly_fit[1] * x_segments[i] + poly_fit[2] * np.square(x_segments[i]) + poly_fit[3] * np.power(x_segments[i], 3)
        poly_error = square_error(y_segments[i], poly_ys)
        # print(poly_error)

        sine_fit = sine_line(x_segments[i], y_segments[i])
        sine_ys = sine_fit[0] + sine_fit[1] * np.sin(x_segments[i])
        sine_error = square_error(y_segments[i], sine_ys)
        # print(sine_error)

        error = min(linear_error, poly_error, sine_error)
        if error == linear_error:
            line[i] = linear
            equation[i] = linear_fit
            print("Linear")
        elif error == poly_error:
            line[i] = polynomial
            equation[i] = poly_fit
            # print("Cubic")
        elif error == sine_error:
            line[i] = sine
            equation[i] = sine_fit
            print("Sine")
        total_error += error
        
    print(total_error)




    # if --plot parameter given, visualise result
    if len(sys.argv) == 3 and sys.argv[2] == "--plot":
        view_data_segments(points[0], points[1], line, equation)
