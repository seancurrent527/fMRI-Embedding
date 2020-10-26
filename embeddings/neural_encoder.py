import numpy as np
import sys, os
import argparse
import tensorflow as tf
sys.path.insert(0, '.')

import data_processing

class AutoEncoder:
    def __init__(self, input_dim, embedding_dim, activation = 'relu'):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        self.encoder_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(embedding_dim, activation = activation, input_shape = (input_dim,)))
        self.decoder_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_dim, activation = 'linear', input_shape = (embedding_dim,)))
        
        encoder_in = tf.keras.layers.Input((None, input_dim))
        encoder_out = self.encoder_layer(encoder_in)
        self.encoder = tf.keras.Model(encoder_in, encoder_out)

        decoder_in = tf.keras.layers.Input((None, embedding_dim))
        decoder_out = self.decoder_layer(decoder_in)
        self.decoder = tf.keras.Model(decoder_in, decoder_out)

    def train(self, X, epochs = 50, learning_rate = 0.01, loss = 'mse'):
        model_in = tf.keras.layers.Input((None, self.input_dim))
        model_encoded = self.encoder(model_in)
        model_out = self.decoder(model_encoded)
        model = tf.keras.Model(model_in, model_out)

        model.compile(loss = loss, optimizer = 'adam')
        model.fit(X, X, epochs = epochs, batch_size = 32)

    def encode(self, data):
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]
            return self.encoder.predict(data)[0]
        return self.encoder.predict(data)

    def decode(self, data):
        return self.decoder.predict(data)

class Transformer:
    def __init__(self, embedding_dim):
        pass

#==================================================
def main():
    X, y = data_processing.read_data('maps_conmat.mat', 'maps_age.mat')
    #X = data_processing.adjacency_matrix(X)

    model = AutoEncoder(X.shape[-1], 64, activation = 'relu')
    model.train(X, epochs = 50, learning_rate = 0.001, loss = 'mse')

if __name__ == '__main__':
    main()