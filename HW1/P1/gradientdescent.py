import loadFittingDataP1, loadParametersP1
import math
import numpy as np

def gradientDescent(f, f_p, guess, step, conv):
    # f: function that we are given
    # f_p: (f' prime) optional function parameter test this function for f being the quadratic bowl and the negative Gaussian function
    # guess: initial guess
    # step: step size
    # conv: convergence criterion

    x_new = guess
    y_new = f(x_new)
    y_old = y_new + 2 * conv
    i = 0  # iterator counts
    while abs(y_old - y_new) >= conv:  # while we have not converged
        x_old = x_new
        y_old = y_new
        x_new = x_old - step * f_p(x_old)
        y_new = f(x_new)
        i += 1
        diff = y_old - y_new
        print "iteration", i
        print "x_i", x_new
        print "y_i", y_new
        print "diff", diff," conv", conv

    return x_new, y_new, i


def gradientDescentFD(f, h, guess, conv):
    # f: function that we are given
    # f_p: (f' prime) optional function parameter test this function for f being the quadratic bowl and the negative Gaussian function
    # guess: initial guess
    # conv: convergence criterion

    x_new = guess
    y_new = f(x_new)
    y_old = y_new + 2 * conv
    i = 0  # iterator counts
    while abs(y_old - y_new) >= conv:  # while we have not converged
        x_old = x_new
        y_old = y_new
        x_new = x_old - finiteDifference(f, h, x_old)
        y_new = f(x_new)
        i += 1
        diff = y_old - y_new
        print "iteration", i
        print "x_i", x_new
        print "y_i", y_new
        print "diff", diff, " conv", conv

    return x_new, y_new


"""Problem 2"""

def finiteDifference(f, h, x):
    # problem 2 of gradient descent
    # f is the function
    # h is our step size, which is a scalar
    # x is where we are evaluation

    # print "\n Print \n"
    # print x
    # print h
    fD = np.zeros(x.shape)
    for i in xrange(x.shape[0]):
        temp = np.zeros(x.shape)
        temp[i] = h
        temp2 = f(x + 0.5 * temp) - f(x - 0.5 * temp)
        fD[i] = temp2
        # print "ITER", i
        # print temp
        # print temp2
    return fD
    # return f(x + 0.5*h) - f(x - 0.5*h)


def quadraticBowl(x):
    # A and b are global variables <- should change
    return float(0.5 * np.transpose(x) * quadBowlA * x - np.transpose(x) * quadBowlb)


def quadraticBowlGrad(x):
    return quadBowlA * x - quadBowlb


def negGaussian(x):
    return -1 / math.sqrt((2 * math.pi) ** 3 * np.linalg.det(cov)) * np.exp(
        -0.5 * np.transpose(x - mu) * np.linalg.inv(cov) * (x - mu))


def negGaussianGrad(x):
    # return -1*negGaussian(x)*np.linalg.inv(cov)*(x-mu)
    return np.linalg.inv(cov) * (x - mu) * -1 * negGaussian(x)


if __name__ == "__main__":
    f = open("bowl_gd.txt", "w")
    print "running gradientDescent.py"

    global quadBowlA
    global quadBowlb
    global mu
    global cov

    (mu, cov, quadBowlA, quadBowlb) = loadParametersP1.getData()
    mu = np.transpose(np.matrix(mu))
    quadBowlb = np.transpose(np.matrix(quadBowlb))
    # data =  mean, Gaussian covariance, A and b for quadratic bowl in order

    # Starting point
    guess = np.transpose(np.matrix([[20, 10]]))  # specify guess here
    step = 0.05  # specify step here
    conv = 0.0001  # specify convergence here
    h = 1

    print "mean", mu
    print "gauss cov", cov
    print "Quadratic Bowl A", quadBowlA
    print "Quadratic Bowl B", quadBowlb

    gauss = negGaussian(mu)
    gaussGrad = negGaussianGrad(cov)

    #x, y, i = gradientDescent(gauss, gaussGrad, guess, 100, 1000)

    x, y, i = gradientDescent(quadraticBowl, quadraticBowlGrad, guess, 0.1, 0.0001)
    print x, y
    # x, y, i = gradientDescent(quadraticBowl, quadraticBowlGrad, guess, 0.01, 0.0001)
    # print x, y, i
    # x, y, i = gradientDescent(quadraticBowl, quadraticBowlGrad, guess, 0.001, 0.0001)
    # print x, y, i
    # x, y, i = gradientDescent(quadraticBowl, quadraticBowlGrad, guess, 0.0001, 0.0001)
    # x, y, i = gradientDescent(quadraticBowl, quadraticBowlGrad, guess, 0.00001, 0.0001)
    # print x, y, i

    # x,y, i = gradientDescent(quadraticBowl, quadraticBowlGrad, guess, step, conv)

    # print "\n\n"

    # # x2,y2 = gradientDescentFD(quadraticBowl, step, guess, conv)
    # x, y, i = gradientDescent(negGaussian, negGaussianGrad, guess, 10e1, 0.0001)
    # print x, y, i
    # x, y, i = gradientDescent(negGaussian, negGaussianGrad, guess, 10e2, 0.0001)
    # print x, y, i
    # x, y, i = gradientDescent(negGaussian, negGaussianGrad, guess, 10e3, 0.0001)
    # print x, y, i
    # x, y, i = gradientDescent(negGaussian, negGaussianGrad, guess, 10e4, 0.0001)
    # print x, y, i
    # # print x2,y2
