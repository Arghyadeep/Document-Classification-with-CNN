## Document-Classification-with-CNN
This repository is a Document Classification system using convolutional neural networks using keras. The architecture is comprised of three key pieces:

Word Embedding: A distributed representation of words where different words that have a similar meaning (based on their usage) also have a similar representation.
Convolutional Model: A feature extraction model that learns to extract salient features from documents represented using a word embedding.
Fully Connected Model: The interpretation of extracted features in terms of a predictive output.

![alt text](https://github.com/Arghyadeep/Document-Classification-with-CNN/blob/master/doc_classification%20cnn.png)

# Dataset: 
The dataset used for this project is BBC news dataset. It can be downloaded from the following link.
http://mlg.ucd.ie/datasets/bbc.html 
It has 5 categories, Business, Technology, Sports, Entertainment and Politics. The classes are well distributed and hence there is no imbalance. The tfidf vectors for every document is converted to a corresponding 2D vector using TSNE to visualize the documents in a 2D space. 

![alt text](https://github.com/Arghyadeep/Document-Classification-with-CNN/blob/master/doc_classifier_bbc.png)

# Process:
After visualization the entire dataset is converted to a CSV file with two columns (documents and labels). The documents are labelled 1 through 5. A standard cleaning process (e.g. removing stopwords, lowering all words, removing punctuations, removing words less than length 2 etc) is performed. Next the data is split into test and train sets. Now the train and test set documents are vectorized using either of these 3 techniques:

1. Keras embeddings (keras deep learning library embedding layer)
2. Word2vec embeddings 
3. Glove vectors

And hence we try to to tune hyperparameters on a validation set and try to get best results for each of these methods and compare them. The different hyperparameters tuned were: 

1. Number of filters
2. kernel size
3. dropout units
4. dense layer units
5. epochs (at times the model worked well with early stopping)
6. activation and loss functions

The ipython notebooks has the best settings after many iterations of trial and error.

# Results:
The trained keras embeddings provided the best validation accuracy of 96.61 percent, followed by glove vectors with 95.13 percent, and word2vec embeddings with 89.19 percent. The confusion matrices are given below for a detailed representation.

confusion matrix for word2vec vectors
![alt text](https://github.com/Arghyadeep/Document-Classification-with-CNN/blob/master/word2vec.png)

confusion matrix for glove vectors
![alt text](https://github.com/Arghyadeep/Document-Classification-with-CNN/blob/master/glove%20vectors.png)

confusion matrix for keras embeddings
![alt text](https://github.com/Arghyadeep/Document-Classification-with-CNN/blob/master/keras.png)



