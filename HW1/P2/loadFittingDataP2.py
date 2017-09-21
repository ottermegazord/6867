import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

def linear_regression(data_X, data_y):

    xTx = X.T.dot(X)
    XtX = np.linalg.inv(xTx)
    XtX_xT = XtX.dot(X.T)
    theta = XtX_xT.dot(y)

    return theta

def getData(ifPlotData=True):
    # load the fitting data and (optionally) plot out for examination
    # return the X and Y as a tuple

    data = pl.loadtxt('curvefittingp2.txt')

    X = data[0,:]
    Y = data[1,:]

    if ifPlotData:
        plt.plot(X,Y,'-o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return (X,Y)

def main():
    X, Y = getData()

if __name__ == "__main__":
    X, Y = getData()
    print X, Y