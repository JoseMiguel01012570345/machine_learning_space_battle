"""
dbscan model

a module to create dbscan models to work with signals
"""
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,homogeneity_score,completeness_score,v_measure_score,adjusted_mutual_info_score,adjusted_rand_score
from signals import get_samples_features_vectors

SCALER = StandardScaler()

class DBSCANModelResult:
    
    """
    the abstraction of a dbscan model trained
    """
    
    _homogeneity = None
    _completness = None
    _v_measure = None
    _ami = None
    _ari = None
    
    def __init__(self,model,data,clusters,tags=None):
        self._model = model
        self._clusters = clusters
        self._score = silhouette_score(data,clusters)
        if tags:
            self._homogeneity = homogeneity_score(tags,clusters)
            self._completness = completeness_score(tags,clusters)
            self._v_measure = v_measure_score(tags,clusters)
            self._ami = adjusted_mutual_info_score(tags,clusters)
            self._ari = adjusted_rand_score(tags,clusters)
            pass
        pass
    
    @property
    def Model(self):
        """
        returns the model
        """
        return self._model
    
    @property
    def AdjustedRandScore(self):
        return self._ari
    
    @property
    def SilhouetteScore(self):
        return self._score
    
    @property
    def Homogeneity(self):
        return self._homogeneity
    
    @property
    def Completness(self):
        return self._completness
    
    @property
    def V_Measure(self):
        return self._v_measure
    
    @property
    def AdjustedMutualInfoScore(self):
        return self._ami
    
    pass

def dbscan_signal_classifier(signals,tags=None,**kwargs):
    """
    returns an instance of LogisticRegressionModelResult class
    """
    eps = 0.5
    min_samples=5
    
    if 'eps' in kwargs.keys():
        eps = kwargs['eps']
        pass
    if 'min_samples' in kwargs.keys():
        min_samples = kwargs['min_samples']
        pass
    
    signals_features = get_samples_features_vectors(signals)
    scaled_data = SCALER.fit_transform(signals_features)
    db = DBSCAN(eps=0.5,min_samples=min_samples)
    clusters = db.fit_predict(scaled_data)
    return DBSCANModelResult(db,scaled_data,clusters,tags)