import numpy as np
import sys, os
import argparse
sys.path.insert(0, '.')

import data_processing
from utils.plot_embeddings import generate_embedding_vis

class MatrixFactorization:
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


def main():
    X, y = data_processing.read_data('maps_conmat.mat', 'maps_age.mat')
    Xm = X.mean(axis = 0)

    factorization = MatrixFactorization(Xm, 2)
    factorization.fit(200, 0.00001)

    generate_embedding_vis(Xm, factorization.factor, embedding_name="Matrix Factorization")

if __name__ == '__main__':
    main()