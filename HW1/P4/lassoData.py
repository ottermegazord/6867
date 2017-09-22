import pdb
import random
import pylab as pl
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values

# converts X to phi feature vectors
def phix(X):
    fv_length = 13
    phi = np.empty([np.shape(X)[0], fv_length])
    phi[:, 0] = X #fills first column with X
    for i in range(1, fv_length):
        phi[:, i] = np.sin(0.4*np.pi*X*i)
    return phi

 # feature_vector_length = 13
 #    phi_X = np.empty([np.shape(X)[0], feature_vector_length])
 #    phi_X[:,0] = X
 #    # phi_X[:,0] = X[:,0]
 #    for i in range(1,feature_vector_length):
 #        # phi_X[:,i] = np.sin(0.4*np.pi*X[:,0]*i)
 #        phi_X[:,i] = np.sin(0.4*np.pi*X*i)
 #    return phi_X


def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def lassoTrainData():
    return getData('lasso_train.txt')

def lassoValData():
    return getData('lasso_validate.txt')

def lassoTestData():
    return getData('lasso_test.txt')

def lassoTrueWData():
    return pl.loadtxt('lasso_true_w.txt')

def main():
    #Load All Data
    train_X, train_Y = lassoTrainData()
    val_X, val_Y = lassoValData()
    test_X, test_Y = lassoTestData()
    trueW = lassoTrueWData()

    trainphiX = phix(train_X[:,0])
    plottingX = np.linspace(-1,1,1000)
    plottingphiX = phix(plottingX)


    # wT.feature vector = y_real
    #print trueW.shape, plottingphiX.shape
    trueY = np.dot(trueW, np.transpose(plottingphiX))

    #train sklearn lasso model classifier with data

    clf = linear_model.Ridge(alpha = .5)
    clf.fit(trainphiX, train_Y)
    ridge_theta = clf.coef_
    y_ridge = np.dot(ridge_theta[:], np.transpose(plottingphiX))
    plt.plot(plottingX, y_ridge[0,:], label = "Ridge")



    A = [0.001, 0.1, 0.5, 5.0, 10.0, 50.0, 100.0]
    for i in range(0, len(A)):
        clf = linear_model.Lasso(alpha=A[i])
        clf.fit(trainphiX, train_Y) #
        lasso_theta = clf.coef_ # outputs lasso's weight parameter
        y_lasso = np.dot(lasso_theta, np.transpose(plottingphiX))
        plt.plot(plottingX, y_lasso, label= 'Alpha: %.2f' %A[i])
    plt.plot(plottingX, trueY,label = "True")
    plt.legend()
    plt.show()

    plottingTrue = np.linspace(1, 13, 13)
    plt.title("W true")
    plt.bar(plottingTrue, trueW)
    plt.show()

    clf = linear_model.Lasso(alpha=0.05)
    clf.fit(trainphiX, train_Y)  #
    lasso_theta = clf.coef_  # outputs lasso's weight parameter
    print lasso_theta

    plottingTrue = np.linspace(1, 13, 13)
    plt.title("estimated w with LASSO Lambda = 0.05")
    plt.bar(plottingTrue, lasso_theta)
    plt.show()

    clf = linear_model.Lasso(alpha=0.00000000001)
    clf.fit(trainphiX, train_Y)  #
    lasso_theta = clf.coef_  # outputs lasso's weight parameter
    print lasso_theta

    plottingTrue = np.linspace(1, 13, 13)
    plt.title("estimated w with LASSO Lambda = 0")
    plt.bar(plottingTrue, lasso_theta)
    plt.show()

    plottingTrue = np.linspace(1, 13, 13)
    a = np.matrix([ridge_theta[0, :]])
    print a
    plt.title("estimated w with ridge")
    plt.bar(plottingTrue, ridge_theta[0, :])
    plt.show()


if __name__ == "__main__":
    main()

