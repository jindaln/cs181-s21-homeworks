import numpy as np
import math

data = [(0., 0., 0.),
        (0., 0.5, 0.),
        (0., 1., 0.),
        (0.5, 0., 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 1., 0.5),
        (1., 0., 1.),
        (1., 0.5, 1.),
        (1., 1., 1.)]

alpha = 10

W1 = alpha * np.array([[1., 0.], [0., 1.]])
W2 = alpha * np.array([[0.1, 0.], [0., 1.]])
W3 = alpha * np.array([[1., 0.], [0., 0.1]])


def compute_prediction(x_i1, x_i2, i, W):
    numerator = 0.
    denominator = 0.
    for n in range(len(data)):
        if (n != i):
            x_n1 = data[n][0]
            x_n2 = data[n][1]
            a_1 = x_n1 - x_i1
            a_2 = x_n2 - x_i2
            K = math.exp(-1 * (a_1**2 * W[0][0]  + 2 * a_1 * a_2 * W[0][1] + a_2**2 * W[1][1])) 
            numerator += K * data[n][2]
            denominator += K
    return numerator/denominator
            

def compute_loss(W):
    loss = 0.
    for i in range(len(data)):
        loss += (data[i][2] - compute_prediction(data[i][0], data[i][1], i, W))**2
    return loss


print(compute_loss(W1))
print(compute_loss(W2))
print(compute_loss(W3))