import numpy as np
import sys, os
import argparse
sys.path.insert(0, '.')

import data_processing
from embeddings.random_walk import random_walk

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim, kernel_width, learning_rate):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernel_width = kernel_width
        self.learning_rate = learning_rate
        self.encoder = np.random.uniform(size = (self.vocab_size, self.embedding_dim))
        self.decoder = np.random.uniform(size = (self.embedding_dim, self.vocab_size))

    def softmax(self, output):
        probabilities = np.exp(output) / np.exp(output).sum()
        return probabilities

    def forward(self, seq):
        embedding = self.encode(seq)
        output = self.decode(embedding)
        probabilities = self.softmax(output)
        return probabilities, embedding, output

    def backprop(self, err, embedding, word):
        grad_decoder = np.outer(embedding, err)
        grad_encoder = np.outer(word, (self.decoder @ err))

        self.encoder -= self.learning_rate * grad_encoder
        self.decoder -= self.learning_rate * grad_decoder

class Skip_Gram(Word2Vec):
    # This is a skip-gram version
    def __init__(self, vocab_size, embedding_dim, kernel_width, learning_rate):
        super().__init__(vocab_size, embedding_dim, kernel_width, learning_rate)

    def encode(self, word):
        embedding = word @ self.encoder
        return embedding

    def decode(self, embedding):
        output = embedding @ self.decoder
        return output

    def error(self, probabilities, context):
        return (probabilities - context).sum(axis = 0)

    def loss(self, output, context):
        return (-output[context.argmax(axis = 1)].sum() + len(context) * np.log(np.sum(np.exp(output))))

    def train(self, X, y, epochs=50):
        for i in range(epochs):
            los = 0
            for word, context in zip(X, y):
                probabilities, embedding, output = self.forward(word)
                err = self.error(probabilities, context)

                self.backprop(err, embedding, word)

                los += self.loss(output, context)

            print(f'Epoch {i + 1}: loss = {los}')

    def train_from_seq(self, seq, epochs=50):
        # seq should be a sequence of one-hot vectors
        ordering = np.random.permutation(len(seq) - (2 * self.kernel_width)) + self.kernel_width
        X, y = [], []
        for i in ordering:
            word = seq[i]
            context = np.concatenate([seq[i - self.kernel_width: i], seq[i + 1: i + 1 + self.kernel_width]])
            X.append(word)
            y.append(context)

        self.train(X, y, epochs=epochs)

class CBOW(Word2Vec):
    # This is a contiuous-bag-of-words version
    def __init__(self, vocab_size, embedding_dim, kernel_width, learning_rate):
        super().__init__(vocab_size, embedding_dim, kernel_width, learning_rate)

    def encode(self, seq):
        embedding = seq @ self.encoder
        return embedding

    def decode(self, seq):
        output = seq @ self.decoder
        return output

    def error(self, probabilities, word):
        return probabilities - word

    def loss(self, output, word):
        return (-output[word == 1] + np.log(np.sum(np.exp(output))))[0]

    def train(self, X, y, epochs=50):
        for i in range(epochs):
            los = 0
            for seq, word in zip(X, y):
                seq = seq.mean(axis=0)
                probabilities, embedding, output = self.forward(seq)
                err = self.error(probabilities, word)

                self.backprop(err, embedding, seq)

                los += self.loss(output, word)

            print(f'Epoch {i + 1}: loss = {los}')

    def train_from_seq(self, seq, epochs=50):
        # seq should be a sequence of one-hot vectors
        ordering = np.random.permutation(len(seq) - (2 * self.kernel_width)) + self.kernel_width
        X, y = [], []
        for i in ordering:
            word = seq[i]
            context = np.concatenate([seq[i - self.kernel_width: i], seq[i + 1: i + 1 + self.kernel_width]])
            X.append(context)
            y.append(word)

        self.train(X, y, epochs=epochs)

#=============================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', type=int)
    args = parser.parse_args()

    X, y = data_processing.read_data('maps_conmat.mat', 'maps_age.mat')
    #X = data_processing.adjacency_matrix(X)

    walk = random_walk(X[0], steps = 1000)
    one_hot = np.zeros((len(walk), 268))
    for i, pos in enumerate(walk):
        one_hot[i, pos] = 1

    if args.version == 0:
        print("Skip-Gram")
        model = Skip_Gram(268, 64, 2, 0.1)
    else:
        print("CBOW")
        model = CBOW(268, 64, 2, 0.1)
    model.train_from_seq(one_hot, epochs = 100)

if __name__ == '__main__':
    main()