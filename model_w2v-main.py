
# coding: utf-8

# In[1]:


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
import nltk
import gensim


# In[2]:


os.chdir("C:\\Users\\Aswathi\\Desktop\\Doc-digit")


# In[3]:


###############################################################################################
# Input the necessary values and locations from the properties.cfg file i.e configuration file
###############################################################################################
df = pickle.load(open('df.pkl', 'rb'))
X = df['Text']
y = df['Target']


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


X


# In[7]:


tokens = [nltk.word_tokenize(sentences) for sentences in X]


# In[8]:


tokens


# In[9]:


len(tokens)


# In[10]:


from gensim.models import Word2Vec


# In[11]:


model = Word2Vec(tokens, min_count=1, size=100)


# In[40]:


print("\n Training the word2vec model...\n")
# reducing the epochs will decrease the computation time
model.train(tokens, total_examples=len(tokens), epochs=150)


# In[77]:


pickle.dump(model, open('model_w2v.pkl', 'wb'))


# In[41]:


vocab = model.wv.vocab.keys()
wordsInVocab = len(vocab)


# In[42]:


model.most_similar('tax')


# In[43]:


import numpy as np
 
def sent_vectorizer(sent, model):
    sent_vec = np.zeros(100)
    numw = 0
    for w in sent:
        try:
            sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
    return list(sent_vec / np.sqrt(sent_vec.dot(sent_vec)))
 
V=[]
for sentence in tokens:
    V.append(sent_vectorizer(sentence, model)) 


# In[44]:


V


# In[45]:


import math
for i in range(0,len(V)):
    for j in range(0,len(V[i])):
        if math.isnan(V[i][j]):
            V[i][j] = 0


# In[46]:


np.isnan(V).any()


# In[47]:


X_train,X_test,y_train,y_test = train_test_split(V,y,test_size=0.3)


# In[48]:


clf_SGD = linear_model.SGDClassifier(class_weight='balanced',loss='log',random_state=100).fit(X_train,y_train)


# In[49]:


y_pred = clf_SGD.predict(X_test)


# In[50]:


print("F1-score: %1.3f" % f1_score(y_test,y_pred,average='macro'))
print("Precision: %1.3f" % precision_score(y_test,y_pred,average = 'macro'))
print("Recall: %1.3f" % recall_score(y_test,y_pred,average = 'macro'))


# In[51]:


from sklearn.ensemble import RandomForestClassifier


# In[52]:


clf_RF = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=100,criterion='gini').fit(X_train,y_train)


# In[53]:


y_pred_RF = clf_RF.predict(X_test)


# In[54]:


print("F1-score: %1.3f" % f1_score(y_test,y_pred_RF,average='macro'))
print("Precision: %1.3f" % precision_score(y_test,y_pred_RF,average = 'macro'))
print("Recall: %1.3f" % recall_score(y_test,y_pred_RF,average = 'macro'))


# In[55]:


from sklearn.ensemble import GradientBoostingClassifier


# In[56]:


clf_GBM = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=100, random_state=100).fit(X_train, y_train)


# In[57]:


y_pred_GBM = clf_GBM.predict(X_test)


# In[58]:


print("F1-score: %1.3f" % f1_score(y_test,y_pred_GBM,average='macro'))
print("Precision: %1.3f" % precision_score(y_test,y_pred_GBM,average = 'macro'))
print("Recall: %1.3f" % recall_score(y_test,y_pred_GBM,average = 'macro'))


# In[59]:


from sklearn import svm


# In[60]:


clf_SVM = svm.SVC(kernel='linear', decision_function_shape='ovr',random_state=100).fit(X_train,y_train)


# In[61]:


y_pred_SVM = clf_SVM.predict(X_test)


# In[62]:


print("F1-score: %1.3f" % f1_score(y_test,y_pred_SVM,average='macro'))
print("Precision: %1.3f" % precision_score(y_test,y_pred_SVM,average = 'macro'))
print("Recall: %1.3f" % recall_score(y_test,y_pred_SVM,average = 'macro'))


# In[63]:


from sklearn.neural_network import MLPClassifier


# In[74]:


clf_MLP = MLPClassifier(solver='adam',activation='relu', alpha=1e-5,hidden_layer_sizes=(400,),learning_rate='constant', random_state=100).fit(X_train,y_train)


# In[75]:


y_pred_MLP = clf_MLP.predict(X_test)


# In[76]:


print("F1-score: %1.3f" % f1_score(y_test,y_pred_MLP,average='macro'))
print("Precision: %1.3f" % precision_score(y_test,y_pred_MLP,average = 'macro'))
print("Recall: %1.3f" % recall_score(y_test,y_pred_MLP,average = 'macro'))


# In[67]:


from sklearn.model_selection import GridSearchCV


# In[68]:


#parameters = {'solver':['adam'], 'activation':['tanh'], 'alpha':[1e-5],'hidden_layer_sizes':[(700,)], 'random_state':[100]}


# In[71]:


parameters = {'solver':['lbfgs', 'sgd', 'adam'],'activation':['identity', 'logistic', 'tanh', 'relu'],'hidden_layer_sizes':[(100,),(400,),(500,),(700,),(1000,)],
            'learning_rate' : ['constant', 'invscaling', 'adaptive']}


# In[72]:


gs_clf_MLP = GridSearchCV(clf_MLP, parameters,n_jobs=-1).fit(X_train,y_train)


# In[73]:


gs_clf_MLP.best_score_
gs_clf_MLP.best_params_

