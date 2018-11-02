# 1) Implementing just the basic linear model using SGD only
# 2) Stratified K-Fold cross validation
# 3) Trying word embedding with word2vec instead of Tfidf weights

###############################################
### Libraries imported (Dependencies)
###############################################
import pandas as pd
import numpy as np
from io import StringIO
import pickle
from sklearn import linear_model
import os
import configparser
from config import config
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
stop = stopwords.words('english')
import gensim
from gensim.models import Word2Vec, word2vec
from sklearn.pipeline import Pipeline

def make_feature_vec(words, model, num_features):
    """
    Average the word vectors for a set of words
    """
    feature_vec = np.zeros((num_features,),dtype="float32")  # pre-initialize (for speed)
    nwords = 0
    index2word_set = set(model.wv.index2word)  # words known to the model

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            feature_vec = np.add(feature_vec,model[word])

    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec

############################
def get_avg_feature_vecs(reviews, model, num_features):
    """
    Calculate average feature vectors for all reviews
    """
    counter = 0
    review_feature_vecs = np.zeros((len(reviews),num_features), dtype='float32')  # pre-initialize (for speed)

    for review in reviews:
        review_feature_vecs[counter] = make_feature_vec(review, model, num_features)
        counter = counter + 1
    return review_feature_vecs

# Load the punkt tokenizer used for splitting reviews into sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
###############################################################################################
# Input the necessary values and locations from the properties.cfg file i.e configuration file
###############################################################################################
config = configparser.ConfigParser()
config.readfp(open('properties.cfg'))
df = pickle.load(open(config.get('dev','df_pickle_file'), 'rb'))
save_pickle = config.get('dev','save_pickle')
log_file = config.get('dev','log_file')
log = open(log_file,'w')
X = df['Text']
y = df['Target']

X.fillna(0,inplace=True)
y.fillna(0,inplace=True)
###########################################################################################
X_train, X_test,y_train, y_test =train_test_split(X,y,random_state=100)
###########################################################################################
train_sentences = []
for text in df['Text']:
    train_sentences += tokenizer.tokenize(text.strip())

############################################################################################
model_name = 'train_model'
# Set values for various word2vec parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 3       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
model = word2vec.Word2Vec(train_sentences, workers=num_workers,
                size=num_features, min_count = min_word_count,
                window = context, sample = downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
model.save(model_name)


############################################################################################

############################################################################################
# calculate average feature vectors for training and test sets
trainDataVecs = get_avg_feature_vecs(X_train, model, num_features)
testDataVecs = get_avg_feature_vecs(X_test, model, num_features)

#############################################################################################

clf_SGD_w2v = linear_model.SGDClassifier(class_weight='balanced',loss='log',random_state=100).fit(trainDataVecs, y_train)
y_pred = clf_SGD_w2v.predict(testDataVecs)

print(classification_report(y_test, y_pred))
