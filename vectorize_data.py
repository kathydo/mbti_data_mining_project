'''
Name:Kathy Do
StudentID: V00819340
Name of Program: data_scrub.py
Purpose: Vectorizes the input data using CountVectorizer, 
Date: February 26, 2018
'''

import csv
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np



mbti_df = pd.read_csv("/Users/kathydo/Documents/School/SENG_474/Project/mbti_data_scrub.csv")
mbti_df['IE'] = np.where(mbti_df['type'].str[0] == 'I', 1, 0)
mbti_df['NS'] = np.where(mbti_df['type'].str[1] == 'N', 1, 0)
mbti_df['TF'] = np.where(mbti_df['type'].str[2] == 'T', 1, 0)
mbti_df['JP'] = np.where(mbti_df['type'].str[3] == 'J', 1, 0)

'''
using out.csv, vocab is size 80122
using mbti.csv, vocab is size 112954

'''

#print mbti_df.shape
#create 4 new columns in the mbti_df to contain binary values target values for the 4 different personality axes
#if the value for the column a data point is 1, the 

#below, the y values are the actual Introversion (1) - Extroversion (0) classification for that data point
#split original data: 70% into X_train, 30% into y_train_IE
X_train, X_test, y_train_IE, y_test_IE = train_test_split(mbti_df['posts'], mbti_df['IE'], test_size=0.3, random_state = 0)
#split test data: 20% of original into test, 10% into val
X_test, X_val, y_test_IE, y_val_IE  = train_test_split(X_test, y_test_IE, test_size=0.33, random_state=0)

#NS AXIS
X_train, X_test, y_train_NS, y_test_NS = train_test_split(mbti_df['posts'], mbti_df['NS'], test_size=0.3, random_state = 0)
X_test, X_val, y_test_NS, y_val_NS  = train_test_split(X_test, y_test_NS, test_size=0.33, random_state=0)

#TF AXIS
X_train, X_test, y_train_TF, y_test_TF = train_test_split(mbti_df['posts'], mbti_df['TF'], test_size=0.3, random_state = 0)
X_test, X_val, y_test_TF, y_val_TF  = train_test_split(X_test, y_test_TF, test_size=0.33, random_state=0)

#JP AXIS
X_train, X_test, y_train_JP, y_test_JP = train_test_split(mbti_df['posts'], mbti_df['JP'], test_size=0.3, random_state = 0)
X_test, X_val, y_test_JP, y_val_JP  = train_test_split(X_test, y_test_JP, test_size=0.33, random_state=0)

#learn vocabulary from text posts
vectorizer = CountVectorizer().fit(mbti_df['posts'])
#print vectorizer.get_feature_names()
print len(vectorizer.get_feature_names())

# summarize. uncomment to view vocabulary
#print(vectorizer.vocabulary_)

#vectorizing - encode each data set as a vector
X_train_vectorized = vectorizer.transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
X_val_vectorized = vectorizer.transform(X_val)
#the y vectors containing the target data is already in vectors with binary values for classification
