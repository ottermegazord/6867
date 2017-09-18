import loadFittingDataP1, loadParametersP1
import math
import numpy as np

def cost(X, y, theta):
    # compute J (cost) given theta
    cost = 0
    for i in xrange(100):
        cost += (X[i] * theta - y[i]) ** 2
    return cost


def updateTheta(X, y, theta_old, step):
    # update theta
    theta_new = np.zeros(10)  # initiate theta_new to be 0s for now
    for j in xrange(10):
        sum_error = 0
        for i in xrange(100):
            sum_error += y[i] - X[i] * theta_old
        theta_new[j] = theta_old[j] + 2 * step * sum_error * X[i, j]
    return theta_new


def gradientDescentBatch(X, y, theta, step, conv):
    # X: given X data
    # y: given y data
    # step: step size
    # conv: convergence criterion
    # goal: solve theta
    theta_new = theta
    cost_new = cost(X, y, theta_new)
    cost_old = cost_new + 2 * conv
    i = 0  # iterator counts
    while abs(cost_old - cost_new) >= conv:  # while we have not converged
        cost_old = cost_new
        theta_old = theta_new
        theta_new = updateTheta(X, y, theta_old, step)
        cost_new = cost(X, y, theta_new)
        i += 1
        diff = cost_old - cost_new
        print "iteration", i
        print "cost", cost_new
        print "diff", diff, " conv", conv

    return theta_new, i


if __name__ == "__main__":
    print "running gradientDescentBatch.py"

    (X, y) = loadFittingDataP1.getData()
    X = np.matrix(X)
    y = np.transpose(np.matrix(y))
    theta = np.zeros((X.shape[1], 1))  # initiate theta at 0
    step = 0.05  # specify step here
    conv = 0.0001  # specify convergence here

    # print X.shape
    # print theta.shape
    # print y.shape

    print cost(X, y, theta)

    #theta = gradientDescentBatch(X,y,theta,step,conv)

    #print "final", theta
