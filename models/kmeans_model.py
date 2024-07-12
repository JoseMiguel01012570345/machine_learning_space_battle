"""
kmeans model

a module to create k-means models to work with signals
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from signals import get_samples_features_vectors

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

def kmeans_signal_classifier(signals,**kwargs):
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
    
    signals_features = get_samples_features_vectors(signals)
    kmeans = KMeans(n_clusters=n_clusters,random_state=42)
    kmeans.fit(signals_features)
    return KMeansModelResult(kmeans,signals_features)