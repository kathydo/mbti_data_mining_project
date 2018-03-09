'''
Name:Kathy Do
StudentID: V00819340
Name of Program: vectorize_data_strat.py
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

df_length =  mbti_df.shape[0]

mbti_df['IE'] = np.where(mbti_df['type'].str[0] == 'I', 1, 0)
mbti_df['NS'] = np.where(mbti_df['type'].str[1] == 'N', 1, 0)
mbti_df['TF'] = np.where(mbti_df['type'].str[2] == 'T', 1, 0)
mbti_df['JP'] = np.where(mbti_df['type'].str[3] == 'J', 1, 0)

y_IE = mbti_df['IE']
y_NS = mbti_df['NS']
y_TF = mbti_df['TF']
y_JP = mbti_df['JP']

num_introverts = np.sum(mbti_df['IE'])
#print num_introverts
print "I:E ratio : I: " + str(float(num_introverts)/df_length) + " E: " + str(float(df_length - num_introverts)/df_length)

num_intuition = np.sum(mbti_df['NS'])
#print num_intuition
print "N:S ratio : N: " + str(float(num_intuition)/df_length) + " S: " + str(float(df_length - num_intuition)/df_length)
num_thinking = np.sum(mbti_df['TF'])
#print num_introverts
print "T:F ratio : T: " + str(float(num_thinking)/df_length) + " F: " + str(float(df_length - num_thinking)/df_length)
num_judging = np.sum(mbti_df['JP'])
#print num_introverts
print "J:P ratio : J: " + str(float(num_judging)/df_length) + " P: " + str(float(df_length - num_judging)/df_length)

'''
running into problem
-so the data is not stratified
-simply splitting the data randommly into train,test,validation
-if we set the stratify parameter, it will look at the target vector, y, 
and randomly split so each subset of data has the same ratio of 1s to 0s

The problem is:
The target vectors have been split on 4 different axes with differen 1:0 ratio
Therefore, the training data will be split differently as well

i'm trying to think this through to see this is actually a problem for our goal:

-Goal: we are trying to prefdict the MBTI personality type for each user
-we are doing this by classifying each dimension (I/E, N/S, T/F, J/P)
-and then combining the results of each classifier to construct the MBTI code

'''

#print mbti_df.shape
#create 4 new columns in the mbti_df to contain binary values target values for the 4 different personality axes
#if the value for the column a data point is 1, the 

#below, the y values are the actual Introversion (1) - Extroversion (0) classification for that data point
#split original data: 70% into X_train, 30% into y_train_IE
X_train_IE, X_test_IE, y_train_IE, y_test_IE = train_test_split(mbti_df['posts'], mbti_df['IE'], test_size=0.3, random_state = 0, stratify = y_IE)
#split test data: 20% of original into test, 10% into val
X_test_IE, X_val_IE, y_test_IE, y_val_IE  = train_test_split(X_test_IE, y_test_IE, test_size=0.33, random_state=0, stratify = y_test_IE)
#print 'IE X_val'
#print X_val
#NS AXIS
X_train_NS, X_test_NS, y_train_NS, y_test_NS = train_test_split(mbti_df['posts'], mbti_df['NS'], test_size=0.3, random_state = 0, stratify = y_NS)
X_test_NS, X_val_NS, y_test_NS, y_val_NS  = train_test_split(X_test_NS, y_test_NS, test_size=0.33, random_state=0, stratify = y_test_NS)
#print 'NS X_val'
#print X_val
#TF AXIS
#X_train_TF, X_test_TF y_train_TF, y_test_TF = train_test_split(mbti_df['posts'], mbti_df['TF'], test_size=0.3, random_state = 0)
#X_test_TF, X_val_TF, y_test_TF, y_val_TF  = train_test_split(X_test_TF, y_test_TF, test_size=0.33, random_state=0)
X_train_TF, X_test_TF, y_train_TF, y_test_TF = train_test_split(mbti_df['posts'], mbti_df['TF'], test_size=0.3, random_state = 0, stratify = y_TF)
X_test_TF, X_val_TF, y_test_TF, y_val_TF  = train_test_split(X_test_TF, y_test_TF, test_size=0.33, random_state=0, stratify = y_test_TF)

#JP AXIS
X_train_JP, X_test_JP, y_train_JP, y_test_JP = train_test_split(mbti_df['posts'], mbti_df['JP'], test_size=0.3, random_state = 0, stratify = y_JP)
X_test_JP, X_val_JP, y_test_JP, y_val_JP  = train_test_split(X_test_JP, y_test_JP, test_size=0.33, random_state=0, stratify = y_test_JP)

#learn vocabulary from text posts
vectorizer = CountVectorizer().fit(mbti_df['posts'])
#print vectorizer.get_feature_names()
print len(vectorizer.get_feature_names())

# summarize. uncomment to view vocabulary
#print(vectorizer.vocabulary_)

#vectorizing - encode each data set as a vector
X_train_vectorized = vectorizer.transform(X_train_IE)
X_test_vectorized = vectorizer.transform(X_test_IE)
X_val_vectorized = vectorizer.transform(X_val_IE)

X_train_vectorized = vectorizer.transform(X_train_NS)
X_test_vectorized = vectorizer.transform(X_test_NS)
X_val_vectorized = vectorizer.transform(X_val_NS)

X_train_vectorized = vectorizer.transform(X_train_TF)
X_test_vectorized = vectorizer.transform(X_test_TF)
X_val_vectorized = vectorizer.transform(X_val_TF)

X_train_vectorized = vectorizer.transform(X_train_JP)
X_test_vectorized = vectorizer.transform(X_test_JP)
X_val_vectorized = vectorizer.transform(X_val_JP)
#the y vectors containing the target data is already in vectors with binary values for classification

'''
Resources used:
https://stackoverflow.com/questions/43095076/scikit-learn-train-test-split-can-i-ensure-same-splits-on-different-datasets

'''



