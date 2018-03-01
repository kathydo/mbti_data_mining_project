'''
Name: Kathy Do
Name of Program: data_scrub.py
Purpose: Vectorizes the input data using CountVectorizer.
Date: February 26, 2018
'''
import csv
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

mbti_df = pd.read_csv("/Users/kathydo/Documents/School/SENG_474/Project/out.csv")

#print mbti_df.shape

mbti_df['IE'] = np.where(mbti_df['type'].str[0] == 'I', 1, 0)
mbti_df['NS'] = np.where(mbti_df['type'].str[1] == 'N', 1, 0)
mbti_df['TF'] = np.where(mbti_df['type'].str[2] == 'T', 1, 0)
mbti_df['JP'] = np.where(mbti_df['type'].str[3] == 'J', 1, 0)

#below, the y values are the actual Introversion (1) - Extroversion (0) classification for that data point
X_train, X_test, y_train, y_test = train_test_split(mbti_df['posts'], mbti_df['IE'], random_state = 0)

vectorizer = CountVectorizer().fit(X_train)
#print vectorizer.get_feature_names()
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(X_train)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())




