import numpy as np
import sys, os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
sys.path.insert(0, '.')

import data_processing
from embeddings.random_walk import random_walk
from utils.plot_embeddings import calculate_distance_matrix, euclidean_distance
from embeddings.neural_encoder import AutoEncoder, Transformer
from embeddings.factorization import MatrixFactorization, TensorFactorization
from embeddings.word2vec import Skip_Gram, CBOW

# this code was written by Sean

def main():
    X, y = data_processing.read_data('Data/conmat_240.mat', 'Data/age_240.mat')
    Xm = X.mean(axis = 0)

    EMBEDDING_DIM = 16

    #Fully-Connected AutoEncoder
    e_x = tf.keras.layers.Input((None, X.shape[-1]))
    e_o = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(EMBEDDING_DIM, activation='tanh'))(e_x)
    e = tf.keras.Model(e_x, e_o)

    d_x = tf.keras.layers.Input((None, EMBEDDING_DIM))
    d_o = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(X.shape[-1], activation='linear'))(d_x)
    d = tf.keras.Model(d_x, d_o)

    ae_model = AutoEncoder(e, d)
    ae_model.train(X, epochs = 50, learning_rate = 0.001, loss = 'mse')

    #Transformer AutoEncoder
    et_x = tf.keras.layers.Input((X.shape[1], X.shape[2]))
    et_o = Transformer(EMBEDDING_DIM, heads=8, activation='tanh')(et_x)
    et = tf.keras.Model(et_x, et_o)

    dt_x = tf.keras.layers.Input((X.shape[1], EMBEDDING_DIM))
    dt_o = Transformer(X.shape[2], heads=8, activation='linear')(dt_x)
    dt = tf.keras.Model(dt_x, dt_o)

    ae_modelt = AutoEncoder(et, dt)
    ae_modelt.train(X, epochs = 100, learning_rate = 0.001, loss = 'mse')

    #Matrix Factorization
    mat_factorization = MatrixFactorization(Xm, EMBEDDING_DIM)
    mat_factorization.fit(200, 0.0001)

    #Tensor Factorization
    tens_factorization = TensorFactorization(X, EMBEDDING_DIM)
    tens_factorization.fit(50)

    walk = random_walk(Xm, steps = 1000)
    one_hot = np.zeros((len(walk), 268))
    for i, pos in enumerate(walk):
        one_hot[i, :] = Xm[pos]

    #Skip-Gram
    skipgram = Skip_Gram(268, EMBEDDING_DIM, 3, 0.1)
    skipgram.train_from_feature_seq(one_hot, epochs = 200)

    #CBOW
    cbow = CBOW(268, EMBEDDING_DIM, 3, 0.1)
    skipgram.train_from_feature_seq(one_hot, epochs = 200)
    
    og_distances = calculate_distance_matrix(X.reshape((len(X), -1)))

    models = {'AutoEncoder': ae_model, 'Transformer': ae_modelt,
              'Matrix Factorization': mat_factorization,
              'Tensor Factorization': tens_factorization,
              'Skip-Gram': skipgram, 'CBOW': cbow}

    model_distances = {}

    for key, mod in models.items():
        x_embed = mod.encode(X)
        model_distances[key] = calculate_distance_matrix(x_embed.reshape((len(x_embed), -1)))

    #plot distances
    plt.matshow(og_distances, cmap='Blues', vmin = 0)
    plt.title('Original Distances')
    plt.savefig('images/og_distance_matrix.png')

    fig, axes = plt.subplots(2, 3)
    i = 0
    for embedding_name, embedding_distances in model_distances.items():
        r, c = i // 3, i % 3
        axes[r, c].matshow(embedding_distances, cmap = 'Blues', vmin = 0)
        axes[r, c].set_title(embedding_name)
        i += 1
    fig.savefig('images/embedding_distances_matrix.png')

if __name__ == '__main__':
    main()