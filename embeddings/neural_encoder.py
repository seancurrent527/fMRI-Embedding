import numpy as np
import sys, os
import argparse
import tensorflow as tf
sys.path.insert(0, '.')

import data_processing
from utils.plot_embeddings import generate_embedding_vis

class AutoEncoder:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def train(self, X, epochs = 50, learning_rate = 0.01, loss = 'mse'):
        model_in = tf.keras.layers.Input(X.shape[1:])
        model_encoded = self.encoder(model_in)
        model_out = self.decoder(model_encoded)
        model = tf.keras.Model(model_in, model_out)

        model.compile(loss = loss, optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate))
        model.fit(X, X, epochs = epochs, batch_size = 32)

    def encode(self, data):
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]
            return self.encoder.predict(data)[0]
        return self.encoder.predict(data)

    def decode(self, data):
        return self.decoder.predict(data)

class Transformer(tf.keras.layers.Layer):
    def __init__(self, units, heads=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Transformer, self).__init__(**kwargs)

        self.units = units
        self.heads = heads
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    def build(self, input_shape):
        # (batch, timesteps, features)
        assert len(input_shape) == 3
        self.seq_length = input_shape[1]
        self.input_size = input_shape[2]
        self.query_kernel = self.add_weight(name = 'query_kernel',
                                            shape = (self.heads, input_shape[-1], self.units),
                                            initializer = self.kernel_initializer,
                                            regularizer = self.kernel_regularizer,
                                            constraint = self.kernel_constraint,
                                            trainable = True)
        self.key_kernel = self.add_weight(name = 'key_kernel',
                                            shape = (self.heads, input_shape[-1], self.units),
                                            initializer = self.kernel_initializer,
                                            regularizer = self.kernel_regularizer,
                                            constraint = self.kernel_constraint,
                                            trainable = True)
        self.value_kernel = self.add_weight(name = 'value_kernel',
                                            shape = (self.heads, input_shape[-1], self.units),
                                            initializer = self.kernel_initializer,
                                            regularizer = self.kernel_regularizer,
                                            constraint = self.kernel_constraint,
                                            trainable = True)
        self.query_bias = self.add_weight(name = 'query_bias',
                                            shape = (self.heads, self.units),
                                            initializer = self.bias_initializer,
                                            regularizer = self.bias_regularizer,
                                            constraint = self.bias_constraint,
                                            trainable = True)
        self.key_bias = self.add_weight(name = 'key_bias',
                                            shape = (self.heads, self.units),
                                            initializer = self.bias_initializer,
                                            regularizer = self.bias_regularizer,
                                            constraint = self.bias_constraint,
                                            trainable = True)
        self.value_bias = self.add_weight(name = 'value_bias',
                                            shape = (self.heads, self.units),
                                            initializer = self.bias_initializer,
                                            regularizer = self.bias_regularizer,
                                            constraint = self.bias_constraint,
                                            trainable = True)
        self.head_kernel = self.add_weight(name = 'head_kernel',
                                            shape = (self.heads * self.units, self.units),
                                            initializer = self.kernel_initializer,
                                            regularizer = self.kernel_regularizer,
                                            constraint = self.kernel_constraint,
                                            trainable = True)
        self.head_bias = self.add_weight(name = 'head_bias',
                                            shape = (self.units,),
                                            initializer = self.bias_initializer,
                                            regularizer = self.bias_regularizer,
                                            constraint = self.bias_constraint,
                                            trainable = True)

        super(Transformer, self).build(input_shape)

    def call(self, inputs):
        inputs = tf.repeat(tf.expand_dims(inputs, axis = 1), self.heads, axis = 1)
        #print('INPUTS:', inputs.shape)
        distribute = lambda x: tf.repeat(tf.expand_dims(x, axis = 1), self.seq_length, axis = 1)
        queries = tf.matmul(inputs, self.query_kernel) + (distribute(self.query_bias) if self.use_bias else 0)
        #print('QUERIES:', queries.shape)
        keys = tf.matmul(inputs, self.key_kernel) + (distribute(self.key_bias) if self.use_bias else 0)
        #print('KEYS:', keys.shape)
        values = tf.matmul(inputs, self.value_kernel) + (distribute(self.value_bias) if self.use_bias else 0)
        #print('VALUES:', values.shape)
        sims = tf.matmul(queries, tf.transpose(keys, (0, 1, 3, 2))) / np.sqrt(self.units)
        #print('SIMS:', sims.shape)
        attentions = tf.nn.softmax(sims, axis = -1)
        #print('ATTENTIONS:', attentions.shape)
        weighted_sims = tf.matmul(attentions, values)
        #print('WEIGHTED SIMS:', weighted_sims.shape)
        flattened_sims = tf.reshape(tf.transpose(weighted_sims, (0, 2, 1, 3)), [-1, self.seq_length, self.heads * self.units])
        #print('FLATTENED SIMS:', flattened_sims.shape)
        outputs = tf.matmul(flattened_sims, self.head_kernel) + (self.head_bias if self.use_bias else 0)
        #print('OUTPUTS:', outputs.shape)
        return self.activation(outputs)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

#==================================================
def main():
    X, y = data_processing.read_data('maps_conmat.mat', 'maps_age.mat')
    Xm = X.mean(axis = 0)

    EMBEDDING_DIM = 8
    ACTIVATION = 'tanh'
    HEADS = 16

    #Fully-Connected AutoEncoder
    e_x = tf.keras.layers.Input((None, X.shape[-1]))
    e_o = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(EMBEDDING_DIM, activation=ACTIVATION))(e_x)
    e = tf.keras.Model(e_x, e_o)

    d_x = tf.keras.layers.Input((None, EMBEDDING_DIM))
    d_o = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(X.shape[-1], activation='linear'))(d_x)
    d = tf.keras.Model(d_x, d_o)

    model = AutoEncoder(e, d)
    model.train(X, epochs = 50, learning_rate = 0.001, loss = 'mse')
    generate_embedding_vis(Xm, model.encode(Xm), embedding_name='Neural Autoencoder')

    #Transformer AutoEncoder
    et_x = tf.keras.layers.Input((X.shape[1], X.shape[2]))
    et_o = Transformer(EMBEDDING_DIM, heads=HEADS, activation=ACTIVATION)(et_x)
    et = tf.keras.Model(et_x, et_o)

    dt_x = tf.keras.layers.Input((X.shape[1], EMBEDDING_DIM))
    dt_o = Transformer(X.shape[2], heads=HEADS, activation='linear')(dt_x)
    dt = tf.keras.Model(dt_x, dt_o)

    modelt = AutoEncoder(et, dt)
    modelt.train(X, epochs = 100, learning_rate = 0.001, loss = 'mse')
    generate_embedding_vis(Xm, modelt.encode(Xm), embedding_name='Neural Transformer')

if __name__ == '__main__':
    main()