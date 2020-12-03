import numpy as np
import sys, os
import argparse
import tensorflow as tf
sys.path.insert(0, '.')

import data_processing

np.random.seed(5523)

class GraphCNN(tf.keras.layers.Layer):
    def __init__(self, output_dim,
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
        super(GraphCNN, self).__init__(**kwargs)

        self.output_dim = output_dim
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
        assert len(input_shape) == 2
        node_shape, filter_shape = input_shape
        self.num_nodes = node_shape[1]
        self.input_dim = node_shape[-1]
        self.num_filters = filter_shape[1]

        #initialize all necessary weights and kernels
        self.kernel = self.add_weight(name = 'kernel',
                                      shape = (self.num_filters * self.input_dim, self.output_dim),
                                      initializer = self.kernel_initializer,
                                      regularizer = self.kernel_regularizer,
                                      constraint = self.kernel_constraint,
                                      trainable = True)
        self.bias = self.add_weight(name = 'bias',
                                    shape = (self.output_dim,),
                                    initializer = self.bias_initializer,
                                    regularizer = self.bias_regularizer,
                                    constraint = self.bias_constraint,
                                    trainable = True)
        super(GraphCNN, self).build(input_shape)

    def call(self, inputs):
        nodes, filters = inputs
        #nodes has shape (batch, nodes, features)
        #filters has shape (batch, filters, nodes, nodes)

        #expand inputs along filter axis
        nodes = tf.repeat(tf.expand_dims(nodes, axis = 1), self.num_filters, 1)        
        #nodes has shape (batch, filters, nodes, features)

        #perform convolution
        conv_op = tf.matmul(filters, nodes)
        conv_op = tf.reshape(tf.transpose(conv_op, (0, 2, 1, 3)), (-1, self.num_nodes, self.num_filters * self.input_dim))
        #conv_op has shape (batch, nodes, filters * features)

        conv_out = tf.matmul(conv_op, self.kernel)
        #conv_out is shape (batch, nodes, output_dim)

        if self.use_bias:
            conv_out = tf.nn.bias_add(conv_out, self.bias)

        return self.activation(conv_out)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        node_shape, _ = input_shape
        return node_shape[:2] + (self.output_dim,)

def graph_nn(num_features):
    n_in = tf.keras.layers.Input((268, num_features))
    e_in = tf.keras.layers.Input((1, 268, 268))

    c_1 = GraphCNN(8, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))([n_in, e_in])
    c_1 = tf.keras.layers.Dropout(0.2)(c_1)
    c_2 = GraphCNN(1, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))([c_1, e_in])
    c_2 = tf.keras.layers.Dropout(0.2)(c_2)

    h_1 = tf.keras.layers.Flatten()(c_2)

    a_out = tf.keras.layers.Dense(1, activation = 'linear')(h_1)

    return tf.keras.models.Model([n_in, e_in], a_out)

#=============================================
def main():
    X, y = data_processing.read_data('maps_conmat.mat', 'maps_age.mat')

    permutation = np.random.permutation(len(X))
    X, y = X[permutation], y[permutation]

    node_features = np.eye(268)[np.newaxis, ...]
    node_features = np.repeat(node_features, len(X), axis = 0)

    edge_features = X[:, np.newaxis, ...]

    model = graph_nn(268)

    X_train, y_train = [node_features[:200], edge_features[:200]], y[:200]
    X_test, y_test = [node_features[200:], edge_features[200:]], y[200:]

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    model.fit(X_train, y_train, epochs = 3000, batch_size = 32)

    predictions = model.predict(X_test)

    print(predictions)
    print(y_test)

    print('MSE:', ((predictions - y_test) ** 2).mean())

if __name__ == '__main__':
    main()