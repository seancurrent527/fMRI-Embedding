# Implemented By Matt Riffle
import numpy as np
import data_processing
from embeddings.word2vec import Skip_Gram
from embeddings.random_walk import random_walk
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Lasso:
    # max_iters: maximum iterations of coordinate descent
    # reg_term: weight of L1-regularization
    def __init__(self, max_iters = 1000, reg_term = 0.5):
        self.max_iters = max_iters
        self.reg_term = reg_term


    def predict(self, instances):
        instances = instances / np.linalg.norm(instances, axis=0)
        return instances.dot(self.weights)

    # Trains a lasso regression model using coordinate descent algorithm
    def train_coordinate_descent(self, X, y):

        # data must be normalized for lasso regression
        X = X / np.linalg.norm(X, axis=0)

        self.num_instances, self.num_features = X.shape

        # initialize weights
        self.weights = np.array([0] * self.num_features)

        iteration = 0
        # keep going until max iteration
        while iteration < self.max_iters:
            # update weights by finding the minimum of the sub-differential w.r.t the current feature
            # of the lasso cost function
            for i in range(self.num_features):
                # predict using current weights
                predicted = np.reshape(X.dot(self.weights), (-1, 1))
                # get current feature vector
                feature_column = np.reshape(X[:, i], (-1, 1))
                # find minimum using piecewise sub-differential
                z_i = feature_column.T.dot(y - predicted + self.weights[i] * feature_column)
                solution = 0
                if z_i > self.reg_term:
                    solution = z_i - self.reg_term
                elif z_i < -self.reg_term:
                    solution = z_i + self.reg_term
                # set weight equal to minimum of sub-differential of cost function
                self.weights[i] = float(solution)

            print("Iteration Number " + str(iteration + 1))
            iteration += 1

def main():
    dimension = 32
    X, y = data_processing.read_data('data/maps_conmat.mat', 'data/maps_age.mat')

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)

    # average matrix over train data
    avg_matrix = X_train.mean(axis=0)

    # generate random walks
    walk = random_walk(avg_matrix, steps=1000)
    seq = np.zeros((len(walk), 268))
    for i, pos in enumerate(walk):
        seq[i, :] = avg_matrix[pos]
    print(seq.shape)

    skipgram = Skip_Gram(268, dimension, 2, 0.1)
    skipgram.train_from_feature_seq(seq, epochs=200)

    embedded_train_matrix = np.zeros((len(X_train), 268 * dimension))
    for i in range(len(X_train)):
        embedding_train = skipgram.encode(X_train[i])
        embedded_train_matrix[i] = np.ndarray.flatten(embedding_train)

    embedded_test_matrix = np.zeros((len(X_test), 268 * dimension))
    for i in range(len(X_test)):
        embedding_test = skipgram.encode(X_test[i])
        embedded_test_matrix[i] = np.ndarray.flatten(embedding_test)

    lasso = Lasso(100, .01)

    lasso.train_coordinate_descent(embedded_train_matrix, y_train)

    predicted = lasso.predict(embedded_test_matrix)
    print(mean_squared_error(y_test, predicted))
if __name__ == '__main__':
    main()