from stadistics import load,BuildMatrix
from models import *
from scipy.fft import fft
# from signals import mat
import numpy as np
import json

shape = 1026,1026

# generacion de datos
# print('building the maps ...')
# maps = [BuildMatrix(value,shape) for value in load(1000).values()]
# noise = [np.random.randint(2,size=shape) for i in range(1000)]


# print('creating the signals and extracting features ...')

# vectors = get_samples_features_vectors(matrixs_array2signals_array(maps))
# SaveSamplesFeatures(vectors)

# vectors = get_samples_features_vectors(matrixs_array2signals_array(noise))
# SaveSamplesFeatures(vectors,False)

vectors = LoadSamplesFeatures()

print('loading the data ...')
data = {}
for tag,feature,name in vectors:
    if tag:
        data[name] = 0,feature
        pass
    else:
        data[name] = 1,feature
        pass
    pass

goods = [(tag,feature) for tag,feature in data.values() if tag == 0]
noise = [(tag,feature) for tag,feature in data.values() if tag == 1]

tags = [v[0] for v in goods] + [v[0] for v in noise]
vectors = [v[1] for v in goods] + [v[1] for v in noise]

tags = np.array(tags)
vectors = np.array(vectors)

print('fiting the KNN model')
knn_model = knn_signal_classifier(vectors,tags)
print('fiting the KMeans model')
kmeans_model = kmeans_signal_classifier(vectors)
print('fiting the DBSCAN model')
dbscan_model = dbscan_signal_classifier(vectors,tags)
print('fiting the SVM model')
svm_model = svm_signal_classifier(vectors,tags)
print('fiting the Logistic Regressor model')
log_model = logistic_regression_signal_classifier(vectors,tags)