# This code runs the integration of the various graph embedding techniques within a standard fMRI predictive modeling technique and examines the influence of motion artifact
# This code is written in Python 3
# example of how to run in terminal: python Embedding_CPM.py -embed none -target age
# embed options: none, CBOW, skip, matrix, tensor 
# target options: age, task

### PREDICTING AGE

# to run standard predictive model using using no embedding techniques to predict age

python Embedding_CPM.py -embed none -target age

# to run standard predictive model using using CBOW embedding to predict age

python Embedding_CPM.py -embed CBOW -target age

# to run standard predictive model using using Skip-gram embedding to predict age

python Embedding_CPM.py -embed skip -target age

# to run standard predictive model using using matrix factorization embedding to predict age

python Embedding_CPM.py -embed matrix -target age

# to run standard predictive model using using tensor factorization embedding to predict age

python Embedding_CPM.py -embed tensor -target age


### PREDICTING TASK PERFORMANCE

# to run standard predictive model using using no embedding techniques to predict age

python Embedding_CPM.py -embed none -target task

# to run standard predictive model using using CBOW embedding to predict age

python Embedding_CPM.py -embed CBOW -target task

# to run standard predictive model using using Skip-gram embedding to predict age

python Embedding_CPM.py -embed skip -target task

# to run standard predictive model using using matrix factorization embedding to predict age

python Embedding_CPM.py -embed matrix -target task

# to run standard predictive model using using tensor factorization embedding to predict age

python Embedding_CPM.py -embed tensor -target task