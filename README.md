# fMRI-Embedding

This project is organized on a module level. All embedding code is organized in the embeddings folder, which are accessed within code scripts using

**from embeddings.<embedding_file> import <embedding_method>**

Specific embeddings can be run using the *write_embedding.py* program, which can run single or multiple embeddings on a set of hyper parameters. Example calls are shown below (with embedding dimension 64 and good hyperparameters):

- **python utils/write_embedding.py --autencoder -t Data/conmat_240.mat -x Data/conmat_240.mat -o Data/embeddings_240.mat -d 64 -a tanh -e 100 -l 0.0001**

- **python utils/write_embedding.py --transformer -t Data/conmat_240.mat -x Data/conmat_240.mat -o Data/embeddings_240.mat -d 64 -n 16 -a tanh -e 100 -l 0.0001**

- **python utils/write_embedding.py --skipgram -t Data/conmat_240.mat -x Data/conmat_240.mat -o Data/embeddings_240.mat -d 64 -s 1000 -w 3 -e 50 -l 0.001**

- **python utils/write_embedding.py --cbow -t Data/conmat_240.mat -x Data/conmat_240.mat -o Data/embeddings_240.mat -d 64 -s 1000 -w 3 -e 50 -l 0.001**

- **python utils/write_embedding.py --matrix-factor -t Data/conmat_240.mat -x Data/conmat_240.mat -o Data/embeddings_240.mat -d 64 -e 100 -l 0.0001**

- **python utils/write_embedding.py --tensor-factor -t Data/conmat_240.mat -x Data/conmat_240.mat -o Data/embeddings_240.mat -d 64 -e 20**

Multiple embedding flags can be used at a time, and there is an additional **--all** tag to run all embeddings under the hyper parameters (it is not recommended to use this, as all methods have different optimal hyperparameters). Note that each embedding method has a different set of required hyperparameters:

- The **autoencoder** must specify the embedding size **-d**, the activation function for the encoder **-a**, the number of training epochs **-e**, and the learning rate **-l**.

- The **transformer** must specify the embedding size **-d**, the number of attention heads **-n**, the activation function for the encoder **-a**, the number of training epochs **-e**, and the learning rate **-l**.

- The **skipgram** must specify the embedding size **-d**, the length of the random walk **-s**, the context size **-w**, the number of training epochs **-e**, and the learning rate **-l**.

- The **cbow** must specify the embedding size **-d**, the length of the random walk **-s**, the context size **-w**, the number of training epochs **-e**, and the learning rate **-l**.

- The **matrix-factor** must specify the embedding size **-d**, the number of training epochs **-e**, and the learning rate **-l**.

- The **tensor-factor** must specify the embedding size **-d** and the number of training epochs **-e**.

Additionally, all methods must specify a training set **-t**, an evaluation set **-x**, and an output file **-o**, all of which are in the .mat format. Data in the training and evaluation set should be under the *full_conmat* key and are expected to have data shapes (nodes, nodes, samples).
