import numpy as np
import data_processing
from embeddings.neural_encoder import AutoEncoder
from embeddings.random_walk import random_walk
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf

#This code was written by Matt and adapted by Sean

def main():
    # dimensions to test
    DIMENSIONS = [64, 32, 16, 8, 4, 2, 1]

    X, y = data_processing.read_data('Data/maps_conmat.mat', 'Data/maps_age.mat')

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)

    # train embeddings for each dimension
    encoders = list()
    for dimension in DIMENSIONS:

        print(str(dimension) + "-D Embedding Training")
        
        e_x = tf.keras.layers.Input((None, 268))
        e_o = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dimension, activation='tanh'))(e_x)
        e = tf.keras.Model(e_x, e_o)

        d_x = tf.keras.layers.Input((None, dimension))
        d_o = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(268, activation='linear'))(d_x)
        d = tf.keras.Model(d_x, d_o)

        model = AutoEncoder(e, d)
        model.train(X_train, epochs = 100, learning_rate = 0.001, loss = 'mse')

        encoders.append((model, dimension))

    # encode train and test data using embeddings, then flatten for prediction
    embedded_train_list = list()
    embedded_test_list = list()
    for model, dim in encoders:
        embedded_train_matrix = np.zeros((len(X_train), 268 * dim))
        for i in range(len(X_train)):
            embedding_train = model.encode(X_train[i])
            embedded_train_matrix[i] = np.ndarray.flatten(embedding_train)
        embedded_train_list.append(embedded_train_matrix)
        embedded_test_matrix = np.zeros((len(X_test), 268 * dim))
        for i in range(len(X_test)):
            embedding_test = model.encode(X_test[i])
            embedded_test_matrix[i] = np.ndarray.flatten(embedding_test)
        embedded_test_list.append(embedded_test_matrix)

    # train prediction models on encoded train data, then test on encoded test data and calculate Mean Squared Error
    lr_error_list = list()
    svr_error_list = list()
    mlp_error_list = list()
    lr_error_list_train = list()
    svr_error_list_train = list()
    mlp_error_list_train = list()
    for i in range(len(embedded_train_list)):
        lr = Ridge(alpha=2).fit(embedded_train_list[i], y_train)
        svr = SVR().fit(embedded_train_list[i], np.reshape(y_train, -1))
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16, 8), learning_rate_init=0.001, max_iter=1000).fit(embedded_train_list[i], np.reshape(y_train, -1))
        predictedLR = lr.predict(embedded_train_list[i])
        predictedSV = svr.predict(embedded_train_list[i])
        predictedMLP = mlp.predict(embedded_train_list[i])
        lr_error = mean_squared_error(predictedLR, y_train)
        svr_error = mean_squared_error(predictedSV, y_train)
        mlp_error = mean_squared_error(predictedMLP, y_train)
        lr_error_list_train.append(lr_error)
        svr_error_list_train.append(svr_error)
        mlp_error_list_train.append(mlp_error)
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
    plt.bar(np.arange(len(lr_error_list_train)), lr_error_list_train, width, label="LinReg")
    plt.bar(np.arange(len(svr_error_list_train)) + width, svr_error_list_train, width, label="SVR")
    plt.bar(np.arange(len(mlp_error_list_train)) + 2 * width, mlp_error_list_train, width, label="MLP")
    plt.ylabel("MSE")
    plt.xlabel("Dimensions")
    plt.title("Autoencoder Mean Squared Error by Embedding Dimension - Train")
    plt.xticks(np.arange(len(svr_error_list)) + width, list(DIMENSIONS))
    plt.legend(loc="best")
    plt.savefig('images/autoencoder_train')
    plt.show()

    width = 0.35
    plt.bar(np.arange(len(lr_error_list)), lr_error_list, width, label="LinReg")
    plt.bar(np.arange(len(svr_error_list)) + width, svr_error_list, width, label="SVR")
    plt.bar(np.arange(len(mlp_error_list)) + 2 * width, mlp_error_list, width, label="MLP")
    plt.ylabel("MSE")
    plt.xlabel("Dimensions")
    plt.title("Autoencoder Mean Squared Error by Embedding Dimension - test")
    plt.xticks(np.arange(len(svr_error_list)) + width, list(DIMENSIONS))
    plt.legend(loc="best")
    plt.savefig('images/autoencoder_test')
    plt.show()


if __name__ == '__main__':
    main()