import numpy as np
import sys, os
import argparse
import tensorly as tl
sys.path.insert(0, '.')

import data_processing
from utils.plot_embeddings import generate_embedding_vis

#This code was implemented by Sean

class MatrixFactorization:
    #This method is good for symmetric matrices
    def __init__(self, data, embedding_dim):
        self.data = data
        self.embedding_dim = embedding_dim
        self.num_entries = len(self.data)
        self.factor = np.random.uniform(size=(self.num_entries, self.embedding_dim))

    def MSE(self):
        #(x - w@w.T) ** 2
        output = self.factor @ self.factor.T
        loss = ((self.data - output) ** 2).mean()
        return loss

    def MSEgrad(self):
        #-4w(x - w@w.T)
        output = self.factor @ self.factor.T
        gradient = -4 * (self.data - output) @ self.factor
        return gradient

    def fit(self, epochs, learning_rate):
        for i in range(epochs):
            self.factor -= learning_rate * self.MSEgrad()
            print(f'Epoch {i}: loss - {self.MSE()}')

    def encode(self, data):
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]
        input_a = tl.tenalg.khatri_rao([self.factor, self.factor])
        target_a = tl.unfold(data, mode=0).T
        return np.squeeze(np.linalg.solve(input_a.T.dot(input_a), input_a.T.dot(target_a)).T)

class TensorFactorization:
    def __init__(self, data, embedding_dim):
        self.data = data
        self.embedding_dim = embedding_dim
        self.num_entries = len(self.data)
        self.embedding_factor = np.random.uniform(size=(self.num_entries, self.embedding_dim))
        #since connectivity matrices are symmetric, we only need one matrix factor
        self.matrix_factor = np.random.uniform(size=(self.data.shape[1], self.embedding_dim))

    def fit(self, epochs):
        #adapted from https://medium.com/@mohammadbashiri93/tensor-decomposition-in-python-f1aa2f9adbf4
        for i in range(epochs):
            # optimize embedding factor
            input_a = tl.tenalg.khatri_rao([self.matrix_factor, self.matrix_factor])
            target_a = tl.unfold(self.data, mode=0).T
            self.embedding_factor = np.linalg.solve(input_a.T.dot(input_a), input_a.T.dot(target_a)).T

            # optimize matrix factor
            input_b = tl.tenalg.khatri_rao([self.embedding_factor, self.matrix_factor])
            target_b = tl.unfold(self.data, mode=1).T
            self.matrix_factor = np.linalg.solve(input_b.T.dot(input_b), input_b.T.dot(target_b)).T

            res_a = np.square(input_a.dot(self.embedding_factor.T) - target_a).mean()
            res_b = np.square(input_b.dot(self.matrix_factor.T) - target_b).mean()
            print(f"Epoch {i}: embedding loss - {res_a}    matrix loss - {res_b}")

    def encode(self, data):
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]
        input_a = tl.tenalg.khatri_rao([self.matrix_factor, self.matrix_factor])
        target_a = tl.unfold(data, mode=0).T
        return np.squeeze(np.linalg.solve(input_a.T.dot(input_a), input_a.T.dot(target_a)).T)


def main():
    X, y = data_processing.read_data('Data/conmat_240.mat', 'Data/age_240.mat')
    Xm = X.mean(axis = 0)

    factorization = MatrixFactorization(Xm, 8)
    factorization.fit(200, 0.0001)

    #generate_embedding_vis(Xm, factorization.factor, embedding_name="Matrix Factorization")
    generate_embedding_vis(X, factorization.encode(X), embedding_name="Matrix Factorization")

    factorization = TensorFactorization(X, 8)
    factorization.fit(50)

    #generate_embedding_vis(Xm, factorization.matrix_factor, embedding_name="Tensor Factorization")
    generate_embedding_vis(X, factorization.encode(X), embedding_name='Tensor Factorization')

if __name__ == '__main__':
    main()