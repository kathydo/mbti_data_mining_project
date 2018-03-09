import csv
import pandas as pd
import numpy as np

npzfile = np.load('/Users/kathydo/Documents/School/SENG_474/Project/Midterm Data/X_data.npz')

#See data set names
#print(npzfile.files)

X_train_IE = npzfile['X_train_IE']
X_test_IE = npzfile['X_test_IE']
X_val_IE = npzfile['X_val_IE']
y_train_IE = npzfile['y_train_IE']
y_test_IE = npzfile['y_test_IE']
y_val_IE = npzfile['y_val_IE']

X_train_NS = npzfile['X_train_NS']
X_test_NS = npzfile['X_test_NS']
X_val_NS = npzfile['X_val_NS']
y_train_NS = npzfile['y_train_NS']
y_test_NS = npzfile['y_test_NS']
y_val_NS = npzfile['y_val_NS']

X_train_TF = npzfile['X_train_TF']
X_test_TF = npzfile['X_test_TF']
X_val_TF = npzfile['X_val_TF']
y_train_TF = npzfile['y_train_TF']
y_test_TF = npzfile['y_test_TF']
y_val_TF = npzfile['y_val_TF']

X_train_JP = npzfile['X_train_JP']
X_test_JP = npzfile['X_test_JP']
X_val_JP = npzfile['X_val_JP']
y_train_JP = npzfile['y_train_JP']
y_test_JP = npzfile['y_train_JP']
y_val_JP = npzfile['y_val_JP']
