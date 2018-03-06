
'''
Name:Kathy Do
StudentID: V00819340
Name of Program: data_clean.py
Purpose: Takes in the MBTI Kaggle Dataset. Prints out an updated csv file
that has the data cleaned (ie. URLs, numbers, etc. removed) This new csv
file will then be used to vectorize and split into our test/train data.
Date: February 26, 2018
'''

import csv
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

mbti_raw_data = pd.read_csv("/Users/kathydo/Documents/School/SENG_474/Project/mbti_1.csv")
#print(mbti_raw_data)
#print(type(mbti_raw_data))
N = mbti_raw_data.shape[0]

print type(mbti_raw_data.iloc[0]['posts'])

#text_urls_removed = re.sub(r'^https?:\/\/.*[\r\n]*', '', text_post)
#https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python

for i in range(0,N):
	text_post = mbti_raw_data.iloc[i]['posts']
	#df2['amount'] = df2['amount'].str.replace('|||', ' ')
	text_urls_removed = re.sub(r'(http\S+)|([0-9]+)|(\|\|\|)', ' ', text_post)
	#text_urls_removed = re.sub(r'[0-9]+', ' ', text_urls_removed)
	#text_urls_removed = re.sub(r'\|\|\|', ' ', text_urls_removed)
	mbti_raw_data.iloc[i]['posts'] = text_urls_removed

mbti_raw_data.to_csv('out.csv')