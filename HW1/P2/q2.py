import loadFittingDataP2
import numpy as np
import matplotlib.pyplot as plt
import math


def poly_phi(X, M, function='polynomial'): # returns phi Vandermonde
    X_hold = np.ones((X.shape[0], M)) #  make X.shape[0] (N) x M ones array (M+1 because 0, 1, 2... M)
    for i in xrange(1, M):
        if (function == 'polynomial'):
            X_hold[:, [i]] = np.matrix(np.power(X, i))
        elif (function == 'cosine'):
            X_hold[:, [i]] = np.matrix(np.cos(math.pi*X*i))
    return X_hold

def poly_theta(X, Y, M, function='polynomial'): # returns weight vector theta,
    phi = poly_phi(X, M + 1, function)
    weight = np.linalg.inv(np.transpose(phi).dot(phi)).dot(phi.transpose()).dot(Y)  # theta = (X^T X)^-1 X^T y
    print weight
    return weight

def linear_regression(theta, phi):
    return phi.T.dot(theta)

def plot_ML(X, Y, M):
    theta = poly_theta(X, Y, M)
    X_basis = np.matrix([np.linspace(0, 1, 100)]).transpose() # generate basis for plots
    X_plot = poly_phi(X_basis, M + 1)
    Y_plot = X_plot.dot(theta)
    plt.plot(X_basis, Y_plot)

def training_plot():
    X, Y = loadFittingDataP2.getData(ifPlotData='False')
    plt.scatter(X, Y, facecolors='none', edgecolors='b')
    x = np.arange(0, 1, 0.001)
    y = np.cos(x* np.pi) + 1.5 * np.cos(2 * np.pi * x)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')

def theta_agrad(X,Y,theta):
    return 2*X.transpose().dot(X.dot(theta) - Y)

def SSE_deriv(X,Y,theta):
    phi_new = poly_phi(X, theta.shape[0])
    new = Y - phi_new.dot(theta)
    SSE = np.sum(np.power(new,2))
    grad = theta_gradient(phi_new, Y, theta)
    norm_grad = np.linalg.norm(grad)
    return SSE, grad, norm_grad

def theta_gradient(X,Y,theta):
    return 2 * X.transpose().dot(X.dot(theta) - Y)

if __name__ == "__main__":
    # Load Parameters
    X_raw, Y_raw = loadFittingDataP2.getData(ifPlotData='False')
    X = np.transpose(np.matrix(X_raw))
    Y = np.transpose(np.matrix(Y_raw))
    M = [0, 1, 3, 10]
    step = 0.01

    for i in range(0, len(M)):
        theta = poly_theta(X, Y, i, function='cosine')
        plot_ML(X, Y, M[i])
        plt.show()
    # print theta.shape
        #print SSE_deriv(X, Y, theta)

    theta = poly_theta(X, Y, 10)

    #
    # for i in xrange(0, len(M)):
    #     plot_ML(X,Y,M[i])
    #     plt.plot(X, Y, 'o')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.show()






