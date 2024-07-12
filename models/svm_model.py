"""
svm model

a module to create svm models to work with signals
"""
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report,roc_auc_score,accuracy_score
from signals import get_samples_features_vectors

SCALER = StandardScaler()

class SVMModelResult:
    
    """
    the abstraction for a trained model of svm classification
    """
    
    def __init__(self,model,x_test,y_test):
        y_pred = model.predict(x_test)
        self._report = classification_report(y_test,y_pred)
        self._score = roc_auc_score(y_test,y_pred)
        self._model = model
        self._accuracy = accuracy_score(y_test,y_pred)
        pass
    
    @property
    def Model(self):
        """
        returns the model
        """
        return self._model
    
    @property
    def AccuracyScore(self):
        """
        returns the accuracy score for this model
        """
        return self._accuracy
    
    @property
    def Score(self):
        """
        returns the roc auc score for this model
        """
        return self._score
    
    def CrossValidationScores(self,X,y,cv):
        """
        returns the cross-validation score result
        cv: the number of cuts for the training set
        """
        X_scaled = SCALER.fit_transform(X)
        return cross_val_score(self._model,X_scaled,y,cv=cv)
    
    def CrossValidationMean(self,X,y,cv):
        """
        returns the mean of the corss-validation result
        cv: the number of cuts for the training set
        """
        return np.mean(self.CrossValidationScores(X,y,cv))
    
    def Report(self):
        """
        prints a classification report for this model
        """
        print(self._report)
        pass
    
    pass

def svm_signal_classifier(signals,tags,**kwargs):
    """
    returns an instance of SVMModelResult class
    """
    test_size = 0.2
    random_state = 42
    kernel = 'linear'
    
    if 'test_size' in kwargs.keys():
        test_size = kwargs['test_size']
        pass
    if 'random_state' in kwargs.keys():
        random_state = kwargs['random_state']
        pass
    if 'kernel' in kwargs.keys():
        kernel = kwargs['kernel']
        pass
    
    signals_features = get_samples_features_vectors(signals)
    X_train,X_test,y_train,y_test = train_test_split(signals_features,tags,test_size=test_size,random_state=random_state)
    X_train_scaled = SCALER.fit_transform(X_train)
    X_test_scaled = SCALER.fit_transform(X_test)
    svm_clf = SVC(kernel=kernel)
    svm_clf.fit(X_train_scaled,y_train)
    return SVMModelResult(svm_clf,X_test_scaled,y_test)