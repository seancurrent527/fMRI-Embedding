"""
Kernel regression as mentioned in Tong He paper.
y = K*alpha + "noise"
K - kernel matrix is the similarity matrix.
c1, c2, c3,.....are vectorized data points i.e, the lower triangular marices of
the 268*268 matrix are converted into vectors (row or column, I used rows).
In the similarity matrix, ith row and jth column is :
K(ci, cj) = pearson correlation coefficent(ci,cj). (scipy.stats is used here)
So, the ith row = K(ci,c1) K(ci,c2)....K(ci,cM)
M is the number of training examples.
K dimensions: M*M
alpha dimensions: M*1
Once K is calculated, it is simply linear regression.
Since we have small dataset, and each element in Kernel matrix is only a scalar,
we can even implement the closed form solution of linear regression.
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import scipy.io as sio
from os.path import dirname, join as pjoin

# Linear regression: Stochastic Gradient descent
def linear_regression(train_data, train_target, regularization_coeff, num_iters, learning_rate, convergence):
    """
    Inputs:
    1. Training data - each data point is a row vector
    2. training data target
    3. regularization_coeff - regularization coefficient
    4. Number of iterations
    5. The cost function used is sum of squares.
    6. learning_rate
    7. convergence criterion - here if change in weights is too little, then stop.
    """
    X = train_data

    # Converting a (m,) dataframe into a 2D array.
    if X.ndim == 1:
        X = X[:,None]
    else:
        X = np.array(X)

    T = train_target
    # Converting a (m,) dataframe into a 2D array.
    if T.ndim == 1:
        T = T[:,None]
    else:
        T = np.array(T)

    if X.shape[0] == T.shape[0]:
        print("The dimensions of training data and target datasets match")
    else:
        print("Training data and training target sizes do not match")
        sys.exit()

    beta = regularization_coeff
    lr = learning_rate

    # Convergence limit:
    cl = convergence

    # initializing weights:
    wi = np.random.rand(X.shape[1], T.shape[1])
    w = wi

    # stochastic gradient descent:
    for i in range(num_iters):
        # randomly shuffle the points:
        shuffler = np.random.permutation(len(T))
        X = X[shuffler]
        T = T[shuffler]

        w_old = w

        for j in range(X.shape[0]):
            x = np.matrix(X[j,:])
            gradient = np.dot(np.outer(x,x),w) + beta*w - np.multiply(T[j],x .T)

            # update w
            w = w - lr*gradient
        # Convergence criterion:
        diff = np.linalg.norm(w - w_old)
        if diff < cl:
            break
        else:
            w = w
            # keep continuing

    w_final = w

    return w_final

"""
Loading data:
"""
mat_fname = pjoin('Data/age_240.mat')
mat_fname2 = pjoin('Data/conmat_240.mat')
mat_fname3 = pjoin('Data/task_240.mat')

mat_contents = sio.loadmat(mat_fname)
mat_contents2 = sio.loadmat(mat_fname2)
mat_contents3 = sio.loadmat(mat_fname3)

Age = mat_contents['age']
Task = mat_contents3['task']
T = np.c_[Age, Task]
X = mat_contents2['full_conmat']

"""
X is a 268*268*244 matrix.
There are 244 datapoints and Each data point is a symmetric matrix of dimensions 268*268.
All the diagonal elements are zero in every data point.
So, only lower triangular or upper triangular matrix is needed/useful.

We are going to convert this 3-D array into a 2-D array.
Each row of the 2-D array represents one single data point.
Dimension of a single vectorized data point is: 0.5*(268*268 - 268)
There are 268^2 elements in each original datapoint. Out of which the diagonals are 0.
So, the lower triangular or upper triangular matrix has half of the remaining values.
"""

# Initialize an empty array of appropriate dimensions:
r = X.shape[0]  #268
c = X.shape[1]  #268
l = int(0.5*(r*c - r)) #r=c, so it works
vec_X = np.empty([X.shape[2], l])
# an empty array is initialized.

for c1 in range(X.shape[2]):
    # for all the 244 data points
    a_trial = []
    # an empty list
    for c2 in range(X.shape[0]):
        # for each of the 268 rows
        for c3 in range(c2):
            # each row, upto the diagonal column
            a_trial.append(X[c2,c3,c1])
    # converting the list a_trial to array
    a_trial = np.array(a_trial)
    # converting the 1D array into a 2D array - a column vector
    a_trial = a_trial[:,None]
    # a_trial represents one data point and is a column vector.
    # taking transpose to create the row vector.
    vec_X[c1,:] = a_trial.T

# vec_X is the vectorized input matrix.
# Generating training and test datasets:
# Randomly shuffle the data points
shuffler = np.random.permutation(len(T))
vec_X = X[shuffler]
T = T[shuffler]

# splitting into test and training sets:
vec_X_train = vec_X[:200,:]
vec_X_test = vec_X[200:, :]
T_train = T[:200, :]
T_test = T[200:, :]


# For kernel regression we need to generate similarity matrix kernel.
# Kernel Regression - generating similarity matrix
import scipy
import scipy.stats
# Initializing the similarity matrix training set:
# Training Similarity matrix is of the shape 200*200 (we have 200 points)
K_sim_train = np.empty([vec_X_train.shape[0], vec_X_train.shape[0]])

for i in range(vec_X_train.shape[0]):
    # ith row - ith data points
    xi = vec_X_train[i,:]
    for j in range(vec_X_train.shape[0]):
        # jth data point
        xj = vec_X_train[j,:]
        # jth column of ith row
        K_sim_train[i,j],_ = scipy.stats.pearsonr(xi,xj)

# As we have 200 training examples and 44 test points,
# the similarity matrix of the test data set will be of the shape 44*200
# 44 rows - representing 44 test datapoints,
# 200 columns for 200 training datapoints
# Read Appendix - 1 of the Tong He paper

K_sim_test = np.empty([vec_X_test.shape[0], vec_X_train.shape[0]])
for i in range(vec_X_test.shape[0]):
    xi = vec_X_test[i,:]
    for j in range(vec_X_train.shape[0]):
        xj = vec_X_train[j,:]
        K_sim_test[i,j],_ = scipy.stats.pearsonr(xi,xj)

# if we need a bias term:
#K_train_b = np.ones(K_sim_train.shape[0])
#K_train_b = K_train_b[:,None]
#K_sim_train = np_c[K_train_b, K_sim_train]

# Similarly for the test set:
#K_test_b = np.ones(K_sim_test.shape[0])
#K_test_b = K_test_b[:,None]
#K_sim_test = np_c[K_test_b, K_sim_test]

# Linear regression stochastic gradient solution:
w = linear_regression(train_data = K_sim_train, train_target = T_train, regularization_coeff = 0.001, num_iters = 1000, learning_rate = 0.05, convergence = 0.001)
# closed form solution:
beta = 0.01
wc = np.dot(np.linalg.inv(np.dot(K_sim_train.T,K_sim_train) + beta*np.identity(K_sim_train.shape[0])),np.dot(K_sim_train.T,T_train))


# Training error:
Y_train = np.dot(K_sim_train, w)
train_error = T_train - Y_train
# Sume squared error:
train_error2 = np.dot(train_error.T, train_error)

# Test error:
Y_test = np.dot(K_sim_test, w)
test_error = T_test - Y_test
# Sume squared error:
test_error2 = np.dot(test_error.T, test_error)
