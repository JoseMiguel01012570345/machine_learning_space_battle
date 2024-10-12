"""
models

a module for create models for the processing of bidimensional binary maps
"""
import numpy as np
from signals import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from models.svm_model import svm_signal_classifier
from models.logistic_model import logistic_regression_signal_classifier
from models.knn_model import knn_signal_classifier
from models.kmeans_model import kmeans_signal_classifier
from models.dbscan_model import dbscan_signal_classifier
from models.decision_tree import decision_tree_classifier , decision_tree_regressor
from signals import get_samples_features_vectors

def matrix2signal(matrix):
    return matrix2canonics_coefs(matrix)

def matrixs_array2signals_array(array):
    for matrix in array:
        yield matrix2signal(matrix)
        pass
    pass