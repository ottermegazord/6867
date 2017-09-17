import pylab as pl
import numpy as np
from math import *
import matplotlib as plt


"""Data Set Analytical Tools"""

# Crates Gauss and Gradient given gaussMean and gaussCov

# def createGaussAndGradient(gaussMean, gaussCov):
#     gaussMean = np.mat(gaussMean[:, np.newaxis])
#     gaussCovInv = np.linalg.inv(gaussCov)
#     gaussNormalizer = -1/(sqrt(2*pi)**gaussMean.size * np.linalg.det(gaussCov))
#     def gauss(x):
#         return gaussNormalizer * exp(-(1/2) *( (x - gaussMean).T * gaussCovInv * (x - gaussMean)))
#     def gaussGrad(x):
#         return - gauss(x) * gaussCovInv * (x - gaussMean)
#     return gauss, gaussGrad

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

    print(createGaussAndGradient(gaussMean, gaussCov))

