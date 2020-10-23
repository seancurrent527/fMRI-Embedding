from scipy.io import loadmat
import numpy as np

def read_data(Xfile, yfile, target_variable = 'age'):
    X = loadmat(Xfile)['full_conmat'].transpose((2, 0, 1))
    y = loadmat(yfile)[target_variable]
    return X.T, y

def adjacency_matrix(X):
    adjacent_X = np.ones((len(X), 268, 268))
    index = 0
    for i in range(268):
        for j in range(i):
            adjacent_X[:, i, j] = X[:, index]
            adjacent_X[:, j, i] = X[:, index]
            index += 1
    return adjacent_X

#============================================
def main():
    X, y = read_data('maps_conmat.mat', 'maps_age.mat')
    #X = adjacency_matrix(X)

    print(X)

if __name__ == '__main__':
    main()