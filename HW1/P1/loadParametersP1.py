import pylab as pl
import numpy as np
from math import *
import matplotlib as plt


"""Data Set Analytical Tools"""

# Creates gauss and gaussGrad (gradient) given gaussMean and gaussCov
def createGaussAndGradient(gaussMean, gaussCov):
    gaussMean = np.mat(gaussMean[:, np.newaxis]) #
    gaussCov_Inv = np.linalg.inv(gaussCov) # Inverses gaussCov
    n = gaussMean.size
    determinantGaussCov = np.linalg.det(gaussCov)
    gaussNormalizer = -1/(sqrt(2*pi)**gaussMean.size * determinantGaussCov)

    # As defined in HW1
    def gauss(x): # f(x)
        return gaussNormalizer * exp(-(1/2)*( (x - gaussMean).T * gaussCov_Inv * (x - gaussMean)))
    def gaussGrad(x):
        return - gauss(x)*gaussCov_Inv*( x - gaussMean)
    return gauss, gaussGrad

# Recursive function that terminates when the difference in objective function value on two successive steps is below
# convergence threshold assigned
def convergenceObjectiveFunction(objective_function, convergence_threshold, max_iti):
    def isConverged(pos, old_pos, iti):
        if iti > max_iti: return True  # Has already converged
        objective = objective_function(pos)
        objective_old_pos = objective_function(old_pos)
        diff = abs(objective - objective_old_pos)
        return diff < convergence_threshold
    return isConverged


# # Outputs position of local mininum
def basicGradientDescent(gradient_function, start_pos, rate, isConverged, output):
    if isinstance(rate, float):
        frate = lambda x: rate
    else:
        frate = rate
    old_pos = None
    pos = start_pos
    iti = 0
    while old_pos is None or not isConverged(pos, old_pos, iti):
        old_pos = pos # updates new position
        iti = iti + 1 # adds 1 to current iti
        pos = gradientIterative(gradient_function, pos, frate(iti))
        output.append((pos, iti))
        if iti % 10 == 0:
            pass
    return pos


# # Recursive function that terminates when the difference in norm on two successive steps is below
# # convergence threshold assigned
# def covergenceGradientFunction(gradient_function, convergence_threshold, max_iti):
#     def isConverged(pos, old_pos, iti):
#         if iti > max_iti: return True # has already converged
#         return np.linalg.norm(gradient_function(pos)) < convergence_threshold
#     return isConverged


# Iterative algorithm: in iteration t + 1
# w^(t+1) = w^t - gradient_t*g(w^t)
def gradientIterative(gradient_t, start, rate):
    return start - rate * gradient_t(start)


def getData():

    # load the parameters for the negative Gaussian function and quadratic bowl function
    # return a tuple that contains parameters for Gaussian mean, Gaussian covariance,
    # A and b for quadratic bowl in order

    data = pl.loadtxt('parametersp1.txt')

    gaussMean = data[0,:]
    gaussCov = data[1:3,:]

    quadBowlA = data[3:5,:]
    quadBowlb = data[5,:]

    return (gaussMean,gaussCov,quadBowlA,quadBowlb)


"""Routine starts here"""

if __name__ == '__main__':

    """Load Parameters"""
    gaussMean, gaussCov, quadBowlA, quadBowlb = getData()
    print gaussMean, gaussCov, quadBowlA, quadBowlb

    """Define variables"""
    # Negative gaussian function threshold
    convCriterionA = 1e-20
    stepsA = 1000

    convCriterionB = 1e-20
    stepsB = 1000

    """Output of Gradient Descent"""
    output = [] #local minimum stores here
    start_matrix = np.mat([[0.], [0.]])

    gauss, gaussGrad = createGaussAndGradient(gaussMean, gaussCov)
    convergedO = convergenceObjectiveFunction(gauss, convCriterionA, stepsA)
    print(convergedO)
    print(basicGradientDescent(gaussGrad, start_matrix, 1e8, convergedO, output))





