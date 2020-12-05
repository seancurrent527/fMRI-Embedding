import numpy as np
import sys, os
import argparse
sys.path.insert(0, '.')

from scipy.io import loadmat, savemat

#this code was written by Sean

#command line to run all:
# python utils/write_embedding.py --autoencoder --transformer --skipgram --cbow --matrix-factor --tensor-factor -t Data/conmat_240.mat -x Data/conmat_240.mat -o Data/embeddings_240.mat -d 16 -s 1000 -n 16 -a tanh -e 200 -l 0.0001 -w 3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--autoencoder',   action='store_true', help='run the AutoEncoder embedding. Requires -d, -a, -e, -l.')
    parser.add_argument('--transformer',   action='store_true', help='run the Transformer embedding. Requires -d, -n, -a, -e, -l.')
    parser.add_argument('--skipgram',      action='store_true', help='run the SkipGram embedding. Requires -d, -s, -w, -e, -l.')
    parser.add_argument('--cbow',          action='store_true', help='run the CBOW embedding. Requires -d, -s, -w, -e, -l.')
    parser.add_argument('--matrix-factor', action='store_true', help='run the Matrix Factorization embedding. Requires -d, -e, -l.')
    parser.add_argument('--tensor-factor', action='store_true', help='run the Tensor Factorization embedding. Requires -d, -e.')
    parser.add_argument('--all',           action='store_true', help='run all embeddings. Requires -d, -s, -w, -n, -a, -e, -l.')
    parser.add_argument('-t', type=str, help='.mat file for the training data.')
    parser.add_argument('-x', type=str, help='.mat file for the eval data.')
    parser.add_argument('-o', '--output', type=str, help='.mat file to write eval embeddings to.')
    parser.add_argument('-d', '--embedding-dim', type=int, help='embedding size.')
    parser.add_argument('-s', '--sentence-length', type=int, help='length of random walk for word2vec embeddings.')
    parser.add_argument('-w', '--window', type=int, help='window size for word2vec embeddings.')
    parser.add_argument('-n', '--heads', type=int, help='number of heads for transformer embedding.')
    parser.add_argument('-a', '--activation', type=str, help='activation function for neural embedding methods.')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to train the embedding.')
    parser.add_argument('-l', '--learning-rate', type=float, help='learning rate used to train embedding method.')
    return parser.parse_args()

def autoencoder(train, evaluate, embedding_dim, activation, epochs, learning_rate):
    import tensorflow as tf
    from embeddings.neural_encoder import AutoEncoder

    e_x = tf.keras.layers.Input((None, train.shape[-1]))
    e_o = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(embedding_dim, activation=activation))(e_x)
    e = tf.keras.Model(e_x, e_o)

    d_x = tf.keras.layers.Input((None, embedding_dim))
    d_o = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(train.shape[-1], activation='linear'))(d_x)
    d = tf.keras.Model(d_x, d_o)

    model = AutoEncoder(e, d)
    model.train(train, epochs = epochs, learning_rate = learning_rate, loss = 'mse')
    return model.encode(evaluate)

def transformer(train, evaluate, embedding_dim, heads, activation, epochs, learning_rate):
    import tensorflow as tf
    from embeddings.neural_encoder import AutoEncoder, Transformer

    et_x = tf.keras.layers.Input((train.shape[1], train.shape[2]))
    et_o = Transformer(embedding_dim, heads=heads, activation=activation)(et_x)
    et = tf.keras.Model(et_x, et_o)

    dt_x = tf.keras.layers.Input((train.shape[1], embedding_dim))
    dt_o = Transformer(train.shape[2], heads=heads, activation='linear')(dt_x)
    dt = tf.keras.Model(dt_x, dt_o)

    modelt = AutoEncoder(et, dt)
    modelt.train(train, epochs = epochs, learning_rate = learning_rate, loss = 'mse')
    return modelt.encode(evaluate)

def skipgram(train, evaluate, embedding_dim, sentence_length, window, epochs, learning_rate):
    from embeddings.random_walk import random_walk
    from embeddings.word2vec import Skip_Gram
    
    Xm = train.mean(axis = 0)
    walk = random_walk(Xm, steps = sentence_length)
    one_hot = np.zeros((len(walk), 268))
    for i, pos in enumerate(walk):
        one_hot[i, :] = Xm[pos]

    model = Skip_Gram(268, embedding_dim, window, learning_rate)
    model.train_from_feature_seq(one_hot, epochs = epochs)
    return model.encode(evaluate)

def cbow(train, evaluate, embedding_dim, sentence_length, window, epochs, learning_rate):
    from embeddings.random_walk import random_walk
    from embeddings.word2vec import CBOW
    
    Xm = train.mean(axis = 0)
    walk = random_walk(Xm, steps = sentence_length)
    one_hot = np.zeros((len(walk), 268))
    for i, pos in enumerate(walk):
        one_hot[i, :] = Xm[pos]

    model = CBOW(268, embedding_dim, window, learning_rate)
    model.train_from_feature_seq(one_hot, epochs = epochs)
    return model.encode(evaluate)

def matrix_factor(train, evaluate, embedding_dim, epochs, learning_rate):
    from embeddings.factorization import MatrixFactorization
    
    Xm = train.mean(axis = 0)

    factorization = MatrixFactorization(Xm, embedding_dim)
    factorization.fit(epochs, learning_rate)
    return factorization.encode(evaluate)

def tensor_factor(train, evaluate, embedding_dim, epochs):
    from embeddings.factorization import TensorFactorization
    
    factorization = TensorFactorization(train, embedding_dim)
    factorization.fit(epochs)
    return factorization.encode(evaluate)

def main():
    args = parse_args()

    train = loadmat(args.t)['full_conmat'].transpose((2, 0, 1))
    evaluate = loadmat(args.x)['full_conmat'].transpose((2, 0, 1))

    embeddings = {}
    
    if args.autoencoder or args.all:
        embeddings['autoencoder'] = autoencoder(train, evaluate, args.embedding_dim, args.activation, args.epochs, args.learning_rate)
    if args.transformer or args.all:
        embeddings['transformer'] = transformer(train, evaluate, args.embedding_dim, args.heads, args.activation, args.epochs, args.learning_rate)
    if args.skipgram or args.all:
        embeddings['skipgram'] = skipgram(train, evaluate, args.embedding_dim, args.sentence_length, args.window, args.epochs, args.learning_rate)
    if args.cbow or args.all:
        embeddings['cbow'] = cbow(train, evaluate, args.embedding_dim, args.sentence_length, args.window, args.epochs, args.learning_rate)
    if args.matrix_factor or args.all:
        embeddings['matrix_factor'] = matrix_factor(train, evaluate, args.embedding_dim, args.epochs, args.learning_rate)
    if args.tensor_factor or args.all:
        embeddings['tensor_factor'] = tensor_factor(train, evaluate, args.embedding_dim, args.epochs)

    savemat(args.output, embeddings)

if __name__ == '__main__':
    main()