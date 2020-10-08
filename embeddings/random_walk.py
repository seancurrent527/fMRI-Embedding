import numpy as np
import sys, os
sys.path.insert(0, '.')

import data_processing

def sample(adjacency, i):
    weights = adjacency[i].copy()
    weights = np.exp(weights) / np.exp(weights).sum() # softmax
    weights[i] = 0
    weights = weights / weights.sum()
    return np.random.choice(np.arange(len(weights)), p=weights)

def random_walk(adjacency, steps = 1000):
    walk = []
    node = np.random.randint(0, len(adjacency) - 1)
    walk.append(node)
    for i in range(steps - 1):
        node = sample(adjacency, node)
        walk.append(node)
    return np.array(walk)

#========================================
def main():
    X, y = data_processing.read_data('fake_data_unique.mat', 'fake_targetvariable.mat')
    X = data_processing.adjacency_matrix(X)

    print(random_walk(X[0], steps = 1000))

if __name__ == '__main__':
    main()