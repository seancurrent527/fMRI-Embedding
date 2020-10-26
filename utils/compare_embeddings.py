import numpy as np
import matplotlib.pyplot as plt
import sys, os
import argparse
sys.path.insert(0, '.')

import data_processing
from embeddings.neural_encoder import AutoEncoder
from embeddings.word2vec import CBOW, Skip_Gram
from embeddings.random_walk import random_walk

def euclidean_distance(v1, v2):
    return ((v1 - v2) ** 2).sum()**0.5

def calculate_distance_matrix(data, distance_function = euclidean_distance):
    #data should be a matrix of shape (nodes, features)
    distances = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(i):
            distances[i, j] = euclidean_distance(data[i], data[j])
            distances[j, i] = distances[i, j]
    return distances

def generate_embedding_vis(og_data, embedding_data, embedding_name = ''):
    og_distances = calculate_distance_matrix(og_data)
    embedding_distances = calculate_distance_matrix(embedding_data)
    fig, axes = plt.subplots(1, 2)
    axes[0].matshow(og_distances, cmap = 'Blues', vmin = 0)
    axes[0].set_title('Original Distances')
    axes[1].matshow(embedding_distances, cmap = 'Blues', vmin = 0)
    axes[1].set_title(embedding_name + ' Distances')
    plt.show()

def generate_embedding_vis_array(data_array, embedding_name_array):
    fig, axes = plt.subplots(2,2)
    for i in range(2):
        for j in range(2):
            distances = calculate_distance_matrix(data_array[i][j])
            axes[i, j].matshow(distances, cmap = 'Blues', vmin = 0)
            axes[i, j].set_title(embedding_name_array[i][j])
    plt.show()

#===============================================
def main():
    X, y = data_processing.read_data('maps_conmat.mat', 'maps_age.mat')
    #X = data_processing.adjacency_matrix(X)

    avg_matrix = X.mean(axis = 0)
    print(avg_matrix.shape)

    model = AutoEncoder(X.shape[-1], 64, activation = 'relu')
    model.train(X, epochs = 200, learning_rate = 0.001, loss = 'mse')
    #generate_embedding_vis(avg_matrix, model.encode(avg_matrix), embedding_name='Neural Autoencoder')

    walk = random_walk(avg_matrix, steps = 1000)
    seq = np.zeros((len(walk), 268))
    for i, pos in enumerate(walk):
        seq[i, :] = avg_matrix[pos]
    print(seq.shape)

    skipgram = Skip_Gram(268, 64, 2, 0.1)
    skipgram.train_from_feature_seq(seq, epochs = 200)
    #generate_embedding_vis(avg_matrix, skipgram.encode(avg_matrix), embedding_name='SkipGram')

    cbow = CBOW(268, 64, 2, 0.1)
    cbow.train_from_feature_seq(seq, epochs = 200)
    #generate_embedding_vis(avg_matrix, cbow.encode(avg_matrix), embedding_name='CBOW')

    distances = [[avg_matrix, model.encode(avg_matrix)], [skipgram.encode(avg_matrix), cbow.encode(avg_matrix)]]
    names = [['Original Distances', 'Autoencoder Distances'], ['SkipGram Distances', 'CBOW Distances']]
    generate_embedding_vis_array(distances, names)

if __name__ == '__main__':
    main()