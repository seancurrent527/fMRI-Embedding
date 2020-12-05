import numpy as np
import data_processing
from embeddings.word2vec import Skip_Gram, CBOW
from embeddings.random_walk import random_walk
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from scipy.io import savemat

#this code was written by Matt

def main():
    # dimensions to test
    DIMENSIONS = [64, 32, 16, 8, 4, 2]

    X, y = data_processing.read_data('Data/maps_conmat.mat', 'Data/maps_age.mat')

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)

    # average matrix over train data
    avg_matrix = X_train.mean(axis=0)

    # generate random walks
    walk = random_walk(avg_matrix, steps=1000)
    seq = np.zeros((len(walk), 268))
    for i, pos in enumerate(walk):
        seq[i, :] = avg_matrix[pos]
    print(seq.shape)

    # train embeddings for each dimension
    skipgrams = list()
    for dimension in DIMENSIONS:

        print(str(dimension) + "-D Embedding Training")
        skipgram = CBOW(268, dimension, 2, 0.1)
        skipgram.train_from_feature_seq(seq, epochs=300)

        skipgrams.append((skipgram, dimension))

    # encode train and test data using embeddings, then flatten for prediction
    embedded_train_list = list()
    embedded_test_list = list()
    for skipgram in skipgrams:
        embedded_train_matrix = np.zeros((len(X_train), 268 * skipgram[1]))
        for i in range(len(X_train)):
            embedding_train = skipgram[0].encode(X_train[i])
            embedded_train_matrix[i] = np.ndarray.flatten(embedding_train)
        embedded_train_list.append(embedded_train_matrix)
        embedded_test_matrix = np.zeros((len(X_test), 268 * skipgram[1]))
        for i in range(len(X_test)):
            embedding_test = skipgram[0].encode(X_test[i])
            embedded_test_matrix[i] = np.ndarray.flatten(embedding_test)
        embedded_test_list.append(embedded_test_matrix)

    # train prediction models on encoded train data, then test on encoded test data and calculate Mean Squared Error
    lr_error_list = list()
    svr_error_list = list()
    mlp_error_list = list()
    for i in range(len(embedded_train_list)):
        savemat(f'Data/cbow_{DIMENSIONS[i]}.mat', {'train':embedded_train_list[i] ,'test':embedded_test_list[i]})
        lr = Ridge().fit(embedded_train_list[i], y_train)
        svr = SVR().fit(embedded_train_list[i], np.reshape(y_train, -1))
        mlp = MLPRegressor(hidden_layer_sizes=(100,)).fit(embedded_train_list[i], np.reshape(y_train, -1))
        print(mlp.loss_)
        predictedLR = lr.predict(embedded_test_list[i])
        predictedSV = svr.predict(embedded_test_list[i])
        predictedMLP = mlp.predict(embedded_test_list[i])
        print(str(embedded_test_list[i].shape[-1] // 268) + "-D Predicted")
        lr_error = mean_squared_error(predictedLR, y_test)
        svr_error = mean_squared_error(predictedSV, y_test)
        mlp_error = mean_squared_error(predictedMLP, y_test)
        lr_error_list.append(lr_error)
        svr_error_list.append(svr_error)
        mlp_error_list.append(mlp_error)

    # plot MSE for different embedding dims and prediction methods
    width = 0.35
    plt.bar(np.arange(len(lr_error_list)), lr_error_list, width, label="LinReg")
    plt.bar(np.arange(len(svr_error_list)) + width, svr_error_list, width, label="SVR")
    plt.bar(np.arange(len(mlp_error_list)) + 2 * width, mlp_error_list, width, label="MLP")
    plt.ylabel("MSE")
    plt.xlabel("Dimensions")
    plt.title("SkipGram Mean Squared Error by Embedding Dimension")
    plt.xticks(np.arange(len(svr_error_list)) + width, list(DIMENSIONS))
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    main()