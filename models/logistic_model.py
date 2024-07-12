"""
logistic model

a module to create logistic regression models to work with signals
"""
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score
from signals import get_samples_features_vectors

class LogisticRegressionModelResult:
    
    """
    the abstraction for a logistic model trained classifier
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
        return cross_val_score(self._model,X,y,cv=cv)
    
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

def logistic_regression_signal_classifier(features,tags,**kwargs):
    """
    returns an instance of LogisticRegressionModelResult class
    """
    test_size = 0.2
    random_state = 42
    max_iter = 1000
    
    if 'test_size' in kwargs.keys():
        test_size = kwargs['test_size']
        pass
    if 'random_state' in kwargs.keys():
        random_state = kwargs['random_state']
        pass
    if 'max_iter' in kwargs.keys():
        max_iter = kwargs['max_iter']
        pass
    
    X_train,X_test,y_train,y_test = train_test_split(features,tags,test_size=test_size,random_state=random_state)
    log_reg = LogisticRegression(max_iter=max_iter)
    log_reg.fit(X_train,y_train)
    return LogisticRegressionModelResult(log_reg,X_test,y_test)