"""
models

a module for create models for the processing of bidimensional binary maps
"""
from signals import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# from sklearn.svm import
import numpy as np

def get_samples_features_vectors(samples):
    """
    returns a matrix where each row is a vector with the features for each sample
    """
    features = [get_signal_features(sample) for sample in samples]
    return np.array([np.array([v.Centroid,v.Bandwith,v.Flatness,v.Average]) for v in features])

def get_kmeans_model(samples,n_clusters=2,random_state=42):
    """
    returns a K-MEANS model fited with the given samples and the silhouette score for the model
    """
    kmeans = KMeans(n_clusters=2,random_state=42)
    features = get_samples_features_vectors(samples)
    kmeans.fit(features)
    return kmeans,silhouette_score(features,kmeans.labels_)

def matrix2signal(matrix):
    return matrix2canonics_coefs(matrix)

def matrixs_array2signals_array(array):
    return [matrix2signal(matrix) for matrix in array]