from __future__ import print_function
import os
import sys
import numpy as np
import re
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.model_selection import train_test_split

## data preprocessing
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
# Lemmatize
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
def pre_process_data(data, remove_stop_words=True):
    list_personality = []
    list_posts = []
    len_data = len(data)
    i=0
    cachedStopWords = stopwords.words("english")
    for row in data.iterrows():
        i+=1
        if i % 500 == 0:
            print("%s | %s rows" % (i, len_data))
        ##### Remove and clean comments
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
        list_posts.append(temp)
    list_posts = np.array(list_posts)
    return list_posts

# import data
mbti_1 = pd.read_csv('mbti_1.csv')
posts = mbti_1.posts
list_posts = pre_process_data(mbti_1, remove_stop_words=True)

# One hot encode labels
from sklearn.preprocessing import LabelBinarizer
labels_16=mbti_1['type'].tolist()
encoder=LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
labels_16=encoder.fit_transform(labels_16)
labels_16=np.array(labels_16)

print(labels_16.shape)
print(labels_16[0])
print(labels_16[1])
mbti_1['IE'] = np.where(mbti_1['type'].str[0] == 'I', 1, 0)
mbti_1['NS'] = np.where(mbti_1['type'].str[1] == 'N', 1, 0)
mbti_1['TF'] = np.where(mbti_1['type'].str[2] == 'T', 1, 0)
mbti_1['JP'] = np.where(mbti_1['type'].str[3] == 'J', 1, 0)
label_IE = np.array(mbti_1['IE'])
label_NS = np.array(mbti_1['NS'])
label_TF = np.array(mbti_1['TF'])
label_JP = np.array(mbti_1['JP'])
print("Binarize MBTI type (IE): \n%s" % label_IE)
print("Binarize MBTI type (NS): \n%s" % label_NS)
print("Binarize MBTI type (TF): \n%s" % label_TF)
print("Binarize MBTI type (JP): \n%s" % label_JP)

BASE_DIR = ''
GLOVE_DIR = "glove.6B"
MAX_SEQUENCE_LENGTH = 500
MAX_NB_WORDS = 2000
EMBEDDING_DIM = 100

# build index mapping words in the embeddings set to their embedding vector
print('Indexing word vectors.')
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.%sd.txt'%str(EMBEDDING_DIM)))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

# prepare text samples and their labels
print('Processing text dataset')
texts = [post.replace("link", "") for post in list_posts] # list of text samples
# list of label ids
print('Found %s texts.' % len(texts))

# vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
print('sequences dim:',len(sequences[2]))
word_index = tokenizer.word_index
print('word_index dim:', len(word_index))
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)


# # split the data into a training set and a testing set
X_train_IE, X_test_IE, y_train_IE, y_test_IE = train_test_split(data, label_IE, test_size=0.2, stratify = label_IE,random_state=0)
X_train_NS, X_test_NS, y_train_NS, y_test_NS = train_test_split(data, label_NS, test_size=0.2, stratify = label_NS,random_state=0)
X_train_TF, X_test_TF, y_train_TF, y_test_TF = train_test_split(data, label_TF, test_size=0.2, stratify = label_TF,random_state=0)
X_train_JP, X_test_JP, y_train_JP, y_test_JP = train_test_split(data, label_JP, test_size=0.2, stratify = label_JP,random_state=0)


# prepare embedding matrix
print('Preparing embedding matrix.')
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# Difine the cnn for binaray classification
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(50, 4, activation='relu')(embedded_sequences)
x = MaxPooling1D(4)(x)
x = Dropout(0.2)(x)
x = Conv1D(50, 4, activation='relu')(x)
x = MaxPooling1D(4)(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
predict_proba = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, predict_proba)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print(model.summary())
print('Training convolutional network on IE.')
model.fit(X_train_IE, y_train_IE,epochs=10, batch_size=200)
loss, accuracy = model.evaluate(X_test_IE,y_test_IE)
print('Accuracy: %f' % (accuracy))
prob_IE = model.predict(data)
print(prob_IE)

print('Training convolutional network on NS.')
model.fit(X_train_NS, y_train_NS,epochs=10, batch_size=200)
loss, accuracy = model.evaluate(X_test_NS,y_test_NS)
print('Accuracy: %f' % (accuracy))
prob_NS = model.predict(data)
print(prob_NS)

print('Training convolutional network on TF.')
model.fit(X_train_TF, y_train_TF,epochs=10, batch_size=200)
loss, accuracy = model.evaluate(X_test_TF,y_test_TF)
print('Accuracy: %f' % (accuracy))
prob_TF = model.predict(data)
print(prob_TF)

print('Training convolutional network on JP.')
model.fit(X_train_JP, y_train_JP,epochs=10, batch_size=200)
loss, accuracy = model.evaluate(X_test_JP,y_test_JP)
print('Accuracy: %f' % (accuracy))
prob_JP = model.predict(data)
print(prob_JP)

new_features = np.column_stack((prob_IE,prob_NS,prob_TF,prob_JP))

# Build a cnn to directly predict 16 classes
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(50, 4, activation='relu')(embedded_sequences)
x = MaxPooling1D(4)(x)
x = Dropout(0.2)(x)
x = Conv1D(50, 4, activation='relu')(x)
x = MaxPooling1D(4)(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
predict_proba = Dense(16, activation='softmax')(x)

model2 = Model(sequence_input, predict_proba)
model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print('Training convolutional network on 16 classes.')
X_train_16, X_test_16, y_train_16, y_test_16 = train_test_split(data, labels_16, test_size=0.2, stratify = labels_16, random_state=0)
model2.fit(X_train_16, y_train_16,epochs=10, batch_size=200)
loss, accuracy = model2.evaluate(X_test_16,y_test_16)
print('Accuracy: %f' % (accuracy))

# use the new_features to predict
from keras.models import Sequential
model3 = Sequential()
model3.add(Dense(50, activation='relu', input_shape=(4,)))
model3.add(Dropout(0.2))
model3.add(Dense(50, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(16, activation='softmax'))
model3.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print('Training convolutional network on 16 classes.')
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(new_features, labels_16, test_size=0.2, stratify = labels_16, random_state=0)
model3.fit(X_train_new, y_train_new,epochs=200, batch_size=200)
loss, accuracy = model3.evaluate(X_test_new,y_test_new)
print('Accuracy: %f' % (accuracy))
