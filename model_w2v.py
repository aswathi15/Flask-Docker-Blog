# 1) Implementing just the basic linear model using SGD only
# 2) Stratified K-Fold cross validation

###############################################
### Libraries imported (Dependencies)
###############################################
import pandas as pd
import numpy as np
from io import StringIO
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn import linear_model
import os
from nltk.corpus import stopwords
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
stop = stopwords.words('english')
###############################################################################################
# Input the necessary values and locations from the properties.cfg file i.e configuration file
###############################################################################################
df = pickle.load(open('df.pkl', 'rb'))
X = df['Text']
y = df['Target']
############################################################################################
skf = StratifiedKFold(n_splits=10,random_state = None)

##############################################################################################
for train_index, test_index in skf.split(X,y):
    print("Train:", train_index, "Validation:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
# Split the dataframe into train and test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
##############################################################################################
# Initialize the classes for tfidf, count vectorizer
# Fit_transform the data
def data_transform(data1,data2):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df = 0, norm ='l2', encoding = 'latin-1', stop_words = stop)
    tf_transformer = tfidf.fit(data1)
    data1_tfidf = tf_transformer.transform(data1)
    data2_tfidf = tf_transformer.transform(data2)
    return data1_tfidf,data2_tfidf,tfidf

###############################################################################################
X_train_tfidf,X_test_tfidf,tfidf= data_transform(X_train,X_test)
clf_SGD = linear_model.SGDClassifier(class_weight='balanced',loss='log',random_state=100).fit(X_train_tfidf, y_train)
y_pred = clf_SGD.predict(X_test_tfidf)


print("F1-score: %1.3f" % f1_score(y_test,y_pred,average='macro'))
print("Precision: %1.3f" % precision_score(y_test,y_pred,average = 'macro'))
print("Recall: %1.3f" % recall_score(y_test,y_pred,average = 'macro'))
##############################################################################################
pickle.dump(tfidf, open('tfidf.pkl','wb'))
pickle.dump(clf_SGD, open('model.pkl','wb'))
