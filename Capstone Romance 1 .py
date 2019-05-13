#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import data from json file
import json
books= []
for line in open('C:\\Users\\girlg\\Documents\\Capstone\\goodreads_books_romance.json', 'r'):
    books.append(json.loads(line))

#Remove book descriptions that are not in english, are blank, or have less than 40 characters.
#TO DO: try different character limits
books_eng=[]
for b in books:
    if b['language_code'] in ['en-GB', 'en-US', 'en-CA', 'eng']:
        if b['description'] not in ['']:
            if len(b['description']) > 40:
                books_eng.append(b)


# In[2]:


#Create a new dictionary with only pertinent keys
books_reduced = []
for b in books_eng:
    test_dic = {
        'text_reviews_count':b['text_reviews_count'],
        'average_rating':b['average_rating'],
        'description':b['description'],
        'ratings_count':b['ratings_count']
    }
    books_reduced.append(test_dic)


# In[3]:


#remove html tags from the descriptions
from bs4 import BeautifulSoup
for b in books_reduced:
    temp_soup = BeautifulSoup(b["description"])
    b["description"] = temp_soup.get_text()


# In[4]:


#tokenize the descriptions, remove stop words and punctuation, and stem and lemmatize description words
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
stopWords = set(stopwords.words('english'))
punctuations= ["?", "!", ".", ",", ";", ":", "'", ")", "(", "...", "-", "--", "$", "`", "``", "%", "&", "#", "@", "*"]
wordnet_lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
filtered_desc = ''
count = 0
for b in books_reduced:    
    sent_tokenize_list = sent_tokenize(b["description"])
    desc_filtered = ''
    count = 0
    for s in sent_tokenize_list:
        words = word_tokenize(s)
        
        for w in words:
            if w not in stopWords and w not in punctuations:
                desc_filtered = desc_filtered + ' ' + wordnet_lemmatizer.lemmatize(ps.stem(w.lower()))
                count = count+1
    b["filtered_desc"] = desc_filtered
    b["desc_word_count"] = count


# In[5]:


#convert dictionary to pandas data frame
import pandas as pd
df_books = pd.DataFrame(books_reduced)
df_books["int_rating_count"] = pd.to_numeric(df_books["ratings_count"])


# In[6]:


#convert ratings_count from a string to an int
#remove books with rating counts more than 3 standard deviations from the mean
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df_books['int_rating_count']))
df_books_o=df_books[(z < 3)]


# In[7]:


#create 2 bins for the rating counts. The bins are of equal size.
labels = [1, 2]
df_books_o['rating_count_group'] = pd.qcut(df_books_o['int_rating_count'], 2, labels=labels)


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np

#y=df_books_o['rating_count_group']
cv = CountVectorizer(ngram_range = (1,2), max_df = 0.65, min_df=0.0)
tfidf = TfidfTransformer()
#X_fit=cv.fit(df_books_o['filtered_desc'])
X_fittrans = cv.fit_transform(df_books_o['filtered_desc'])
X_fittrans = tfidf.fit_transform(X_fittrans)


# In[9]:



from scipy.sparse import hstack
X_dtm = hstack((X_fittrans,np.array(df_books_o['desc_word_count'])[:,None]))
X_dtm.tocsr()

#new_desc_count = np.reshape(df_books_o['desc_word_count'].as_matrix(),(len(df_books_o),1))
#.shape(new_desc_count)
#X_withlen = np.concatenate([np.asarray(X_fittrans), new_desc_count], axis=1)                
#np.shape(X_fittrans)
#np.shape(X_withlen)


# In[10]:


#create testing and training sets, X is hte description and y is the rating count bin
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_dtm, df_books_o['rating_count_group'], test_size=0.3, random_state = 42)


# In[11]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)


# In[12]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

results = confusion_matrix(y_test, y_pred_lr)
print ('Confusion Matrix for Logistic Regression:')
print(results) 
print ('Accuracy Score :',accuracy_score(y_test, y_pred_lr))
print ('Report : ')
print (classification_report(y_test, y_pred_lr))


# In[13]:


#classifier 1 Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

clf = MultinomialNB()

train = clf.fit(X_train, y_train)
y_pred_nb = clf.predict(X_test)


# In[14]:



results = confusion_matrix(y_test, y_pred_nb)
print ('Confusion Matrix for Naive Bayes:')
print(results) 
print ('Accuracy Score :',accuracy_score(y_test, y_pred_nb))
print ('Report : ')
print (classification_report(y_test, y_pred_nb))


# In[15]:


#Classifier 2 Linear SVC
from sklearn.svm import LinearSVC
clf = LinearSVC()

train = clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)


# In[16]:



results = confusion_matrix(y_test, y_pred_svc)
print ('Confusion Matrix for SVC:')
print(results) 
print ('Accuracy Score :',accuracy_score(y_test, y_pred_svc))
print ('Report : ')
print (classification_report(y_test, y_pred_svc))


# In[17]:


#classifier 3 SGD Classifier with result
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn import linear_model

sgd = linear_model.SGDClassifier()

sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)


# In[18]:



results = confusion_matrix(y_test, y_pred_sgd)
print ('Confusion Matrix for SGD:')
print(results) 
print ('Accuracy Score :',accuracy_score(y_test, y_pred_sgd) )
print ('Report : ')
print (classification_report(y_test, y_pred_sgd))


# In[23]:


from sklearn.metrics import roc_auc_score

print("Logistic Reg. AUC: ", roc_auc_score(y_test.tolist(), y_pred_lr.tolist()))
print("Naive Bayes AUC: ", roc_auc_score(y_test.tolist(), y_pred_nb.tolist()))
print("SVC AUC:  ", roc_auc_score(y_test.tolist(), y_pred_svc.tolist()))
print("SGD AUC: ", roc_auc_score(y_test.tolist(), y_pred_sgd.tolist()))


# In[27]:


from sklearn.metrics import log_loss

print("Logistic Reg. Log-Loss: ", log_loss(y_test.tolist(), y_pred_lr.tolist()))
print("Naive Bayes Log-Loss: ", log_loss(y_test.tolist(), y_pred_nb.tolist()))
print("SVC Log-Loss:  ", log_loss(y_test.tolist(), y_pred_svc.tolist()))
print("SGD Log-Loss: ", log_loss(y_test.tolist(), y_pred_sgd.tolist()))


# In[11]:


from sklearn.model_selection import train_test_split
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(df_books_o['filtered_desc'], df_books_o['rating_count_group'],
                                                                test_size=0.3, random_state = 42)

vectorizer = CountVectorizer(ngram_range = (1,2), max_df = 0.65, min_df=0.0, max_features=5000)
X_train_onehot = vectorizer.fit_transform(X_train_nn)

from keras.models import Sequential
from keras.layers import Dense
 
model = Sequential()
 
model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train_onehot, y_train_nn, 
          epochs=2, batch_size=128, verbose=1, 
          validation_data=(X_train_onehot, y_train_nn))
y_pred_nn = model.predict_classes(vectorizer.transform(X_test_nn))


# In[15]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

results = confusion_matrix(y_test_nn, y_pred_nn)
print ('Confusion Matrix for NN:')
print(results) 
print ('Accuracy Score :',accuracy_score(y_test_nn, y_pred_nn) )
print ('Report : ')
print (classification_report(y_test_nn, y_pred_nn))


print("NN AUC: ", roc_auc_score(y_test_nn.tolist(), y_pred_nn.tolist()))
print("NN Log-Loss: ", log_loss(y_test_nn.tolist(), y_pred_nn.tolist()))


# In[ ]:




