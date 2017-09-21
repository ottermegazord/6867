import loadFittingDataP2
import numpy as np
import matplotlib.pyplot as plt
import math


def poly_phi(X, M): # returns phi
    X_hold = np.ones((X.shape[0], M + 1)) #  make X.shape[0] (N) x M ones array (M+1 because 0, 1, 2... M)
    for i in xrange(1, M + 1):
        X_hold[:, [i]] = np.matrix(np.power(X, i))

    return X_hold

def poly_theta(X, Y, M): # returns weight vector theta
    phi = poly_phi(X, M)
    weight = np.linalg.inv(np.transpose(phi).dot(phi)).dot(phi.transpose()).dot(Y)  # theta = (X^T X)^-1 X^T y
    return weight

def plot_ML(X, Y, M):
    theta = poly_theta(X, Y, M)
    X_basis = np.matrix([np.linspace(0, 1, 100)]).transpose() # generate basis for plots
    X_plot = poly_phi(X_basis, M)
    Y_plot = X_plot.dot(theta)
    plt.plot(X_basis, Y_plot)

def training_plot():
    X, Y = loadFittingDataP2.getData()
    plt.plot(X, Y, '-o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    # Load Parameters
    # X_raw, Y_raw = loadFittingDataP2.getData()
    # X = np.transpose(np.matrix(X_raw))
    # Y = np.transpose(np.matrix(Y_raw))
    # M = [0, 1, 3, 10]

    # for i in xrange(0, len(M)):
    #     plot_ML(X,Y,M[i])
    #     plt.plot(X, Y, 'o')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.show()

    X_basis = np.matrix([np.linspace(0, 1, 10000)])
    Y_real = np.cos(X_basis * np.pi) + 1.5 * np.cos(2 * np.pi * X_basis)
    print Y_real
    # plt.plot(X_basis, Y_real, '-', lw = 10)
    # plt.show()




