"""
kmeans model

a module to create k-means models to work with signals
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from signals import get_samples_features_vectors
from sklearn.preprocessing import StandardScaler

class KMeansModelResult:
    
    """
    the abstraction of a kmeans model trained
    """
    
    def __init__(self,model,data):
        self._model = model
        self._silhouette_score = silhouette_score(data,model.labels_)
        pass
    
    @property
    def Model(self):
        """
        returns the model
        """
        return self._model
    
    @property
    def SilhouetteScore(self):
        return self._silhouette_score
    
    pass

def kmeans_signal_classifier(features,**kwargs):
    """
    returns an instance of LogisticRegressionModelResult class
    """
    random_state = 42
    n_clusters = 2
    
    if 'random_state' in kwargs.keys():
        random_state = kwargs['random_state']
        pass
    if 'n_clusters' in kwargs.keys():
        n_clusters = kwargs['n_clusters']
        pass
    
    kmeans = KMeans(n_clusters=n_clusters,random_state=42)
    kmeans.fit(features)
    return KMeansModelResult(kmeans,features)