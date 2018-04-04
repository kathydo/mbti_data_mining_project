import csv
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import sparse, io

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


X_train_IE = load_sparse_csr('X_train_IE.npz')
X_test_IE = load_sparse_csr('X_test_IE.npz')
X_val_IE = load_sparse_csr('X_val_IE.npz')
X_train_NS = load_sparse_csr('X_train_NS.npz')
X_test_NS = load_sparse_csr('X_test_NS.npz')
X_val_NS = load_sparse_csr('X_val_NS.npz')
X_train_TF = load_sparse_csr('X_train_TF.npz')
X_test_TF = load_sparse_csr('X_test_TF.npz')
X_val_TF = load_sparse_csr('X_val_TF.npz')
X_train_JP = load_sparse_csr('X_train_JP.npz')
X_test_JP = load_sparse_csr('X_test_JP.npz')
X_val_JP = load_sparse_csr('X_val_JP.npz')
