#####################
# CS 181, Spring 2021
# Homework 1, Problem 2
# Start Code
##################

import math
import matplotlib.cm as cm

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Read from file and extract X and y
df = pd.read_csv('data/p2.csv')

X_df = df[['x1', 'x2']]
y_df = df['y']

X = X_df.values
y = y_df.values

print("y is:")
print(y)
print("x is:")
print(X)

def kernel_w1(x, target, alpha):
    return math.exp(-(alpha * (x[0] - target[0])**2 + alpha * (x[1] - target[1])**2))

def predict_kernel_single_pt(target_x, alpha):
    numerator = 0
    denominator = 0
    for i in range(len(X)):
        if X[i][0] != target_x[0] or X[i][1] != target_x[1]:
            numerator += kernel_w1(X[i], target_x, alpha) * y[i]
            denominator += kernel_w1(X[i], target_x, alpha)
    return (numerator/denominator)

def average_of_ys(indices):
    output = 0
    for i in indices:
        print("This is y\n", y)
        # print("This is the knn y predic \n", y[int(i[0])])
        output += y[int(i[0])]
    # print("This is the output", output, len(indices), "\n")
    return (output/len(indices))

def predict_knn_single_pt(target_x, k):
    unsorted = np.empty([13, 2])
    for i in range(len(X)):
        if X[i][0] != target_x[0] or X[i][1] != target_x[1]:
            unsorted[i][0] = i
            unsorted[i][1] = kernel_w1(X[i], target_x, 1)
    sorted = unsorted[np.argsort(unsorted[:, 1])]
    # print("This is the sortd list \n", sorted)
    sorted_head = sorted[len(X) - 1 - k:, 0:1]
    print("These are sorted \n", sorted_head)
    return average_of_ys(sorted_head)

def predict_kernel(alpha):
    """Returns predictions using kernel-based predictor with the specified alpha."""
    predicted_ys = np.empty([13,])
    for i in range(len(X)):
        predicted_ys[i] = predict_kernel_single_pt(X[i], alpha)
    # print("These are the predictions \n", predicted_ys)
    return predicted_ys

def predict_knn(k):
    """Returns predictions using KNN predictor with the specified k."""
    predicted_ys = np.empty([13,])
    for i in range(len(X)):
        predicted_ys[i] = predict_knn_single_pt(X[i], k)
    # print("These are the predicted values \n", predicted_ys)
    return predicted_ys

# def plot_kernel_preds(alpha):
#     title = 'Kernel Predictions with alpha = ' + str(alpha)
#     plt.figure()
#     plt.title(title)
#     plt.xlabel('x1')
#     plt.ylabel('x2')
#     plt.xlim((0, 1))
#     plt.ylim((0, 1))

#     plt.xticks(np.arange(0, 1, 0.1))
#     plt.yticks(np.arange(0, 1, 0.1))
#     y_pred = predict_kernel(alpha)
#     # print("This is ypred \n", y_pred)
#     # print('L2: ' + str(sum((y - y_pred) ** 2)))
#     norm = c.Normalize(vmin=0.,vmax=1.)
#     plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
#     for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
#         plt.annotate(str(round(y_, 2)),
#                      (x_1, x_2), 
#                      textcoords='offset points',
#                      xytext=(0,5),
#                      ha='center') 

#     # Saving the image to a file, and showing it as well
#     plt.savefig('alpha' + str(alpha) + '.png')
#     plt.show()

def plot_knn_preds(k):
    title = 'KNN Predictions with k = ' + str(k)
    plt.figure()
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    y_pred = predict_knn(k)
    # print(y_pred)
    # print('L2: ' + str(sum((y - y_pred) ** 2)))
    norm = c.Normalize(vmin=0.,vmax=1.)
    plt.scatter(df['x1'], df['x2'], c=y_pred, cmap='gray', vmin=0, vmax = 1, edgecolors='b')
    for x_1, x_2, y_ in zip(df['x1'].values, df['x2'].values, y_pred):
        plt.annotate(str(round(y_, 2)),
                     (x_1, x_2), 
                     textcoords='offset points',
                     xytext=(0,5),
                     ha='center') 
    # Saving the image to a file, and showing it as well
    plt.savefig('k' + str(k) + '.png')
    plt.show()

# for alpha in (0.1, 3, 100):
    # TODO: Print the loss for each chart.
    # plot_kernel_preds(alpha)

for k in (1, 5, len(X)-1):
    # TODO: Print the loss for each chart.
    plot_knn_preds(k)
