import numpy as np
import data_processing
from embeddings.factorization import TensorFactorization
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from scipy.io import savemat

#This code was written by Matt and adapted by Sean

def main():
    # dimensions to test
    DIMENSIONS = [64, 32, 16, 8, 4, 2, 1]

    #8 does not work at all
    #DIMENSIONS = [64, 32, 16, 4, 2]

    X, y = data_processing.read_data('Data/maps_conmat.mat', 'Data/maps_age.mat')

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)

    # train embeddings for each dimension
    encoders = list()
    for dimension in DIMENSIONS:

        print(str(dimension) + "-D Embedding Training")
        
        model = TensorFactorization(X_train, dimension)
        model.fit(50)

        encoders.append((model, dimension))

    # encode train and test data using embeddings, then flatten for prediction
    embedded_train_list = list()
    embedded_test_list = list()
    for model, dim in encoders:
        embedded_train_list.append(model.encode(X_train))
        embedded_test_list.append(model.encode(X_test))

    # train prediction models on encoded train data, then test on encoded test data and calculate Mean Squared Error
    lr_error_list = list()
    svr_error_list = list()
    mlp_error_list = list()
    lr_error_list_train = list()
    svr_error_list_train = list()
    mlp_error_list_train = list()
    for i in range(len(embedded_train_list)):
        savemat(f'Data/tensor_{DIMENSIONS[i]}.mat', {'train':embedded_train_list[i] ,'test':embedded_test_list[i]})
        lr = Ridge(alpha=0.01).fit(embedded_train_list[i], y_train)
        svr = SVR(C=2).fit(embedded_train_list[i], np.reshape(y_train, -1))
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
        print("Target:", np.squeeze(y_test)[:5])
        print("Predicted (LR):", np.squeeze(predictedLR)[:5])
        print("Predicted (SV):", np.squeeze(predictedSV)[:5])
        print("Predicted (MLP):", np.squeeze(predictedMLP)[:5])
        print(str(DIMENSIONS[i]) + "-D Predicted")
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