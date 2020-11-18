import numpy as np
import matplotlib.pyplot as plt
import sys, os

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