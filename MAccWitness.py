import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, RegressorMixin

class MAccWitness(BaseEstimator, RegressorMixin):

    def __init__(self, gamma=1, degree = 1, coef0 = 0, metric='rbf'):
        """
        error regression using kernel
        metric: 'rbf', 'linear', 'poly', 'sigmoid'
        rbf parameters: gamma
        poly parameters: degree, coef0, gamma
        sigmoid parameters: coef0, gamma
        linear parameters: None
        """
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.metric = metric
        self.normalizing_constant = 1
        self.X_ = None # D_0
        self.y_ = None # error on D_0 wrt a model
        self.n_ = None # number of samples in D_0

    def fit(self, X, y=None):
        """
        prepare data for regression of error
        using all training samples to define witness function
        """
        self.X_ = X
        self.y_ = y.reshape(len(y),1)
        self.n_ = len(self.y_)

        # compute normalizing constant
        if self.metric == 'rbf':
            K = pairwise_kernels(X,X,metric=self.metric,gamma=self.gamma)
        elif self.metric == 'linear':
            K = pairwise_kernels(X,X,metric=self.metric)
        elif self.metric == 'poly':
            K = pairwise_kernels(X,X,metric=self.metric, degree=self.degree, coef0=self.coef0, gamma=self.gamma)
        elif self.metric == 'sigmoid':
            K = pairwise_kernels(X,X,metric=self.metric, coef0=self.coef0, gamma=self.gamma)
        else:
            raise ValueError('metric not supported')     
        self.normalizing_constant = np.sqrt(self.y_.T @ K @ self.y_)[0][0]
        # print("normalizing constant: ", self.normalizing_constant)
        return self
    
    def predict(self, X, y=None):
        """
        making prediction of error on new data (y is error here)
        X: N * 512
        X_: M * 512
        K: N * M
        y_: M * 1
        prod: N * 1d
        """
        # compute the cos similarity between X and training samples
        if self.metric == 'rbf':
            K = pairwise_kernels(X,self.X_,metric=self.metric,gamma=self.gamma)
        elif self.metric == 'linear':
            K = pairwise_kernels(X,self.X_,metric=self.metric)
        elif self.metric == 'poly':
            K = pairwise_kernels(X,self.X_,metric=self.metric, degree=self.degree, coef0=self.coef0, gamma=self.gamma)
        elif self.metric == 'sigmoid':
            K = pairwise_kernels(X,self.X_,metric=self.metric, coef0=self.coef0, gamma=self.gamma)
        else:
            raise ValueError('metric not supported')
        prod = np.ravel(K@self.y_)/self.normalizing_constant
        return prod

    def score(self, X, y=None):
        return(np.abs(np.corrcoef(self.predict(X),y)[0,1]))

    def compute_KME(self, X, y_test, y):
        """
        compute KME for a given dataset
        """
        if self.metric == 'rbf':
            K = pairwise_kernels(X,self.X_,metric=self.metric,gamma=self.gamma)
        elif self.metric == 'linear':
            K = pairwise_kernels(X,self.X_,metric=self.metric)
        elif self.metric == 'poly':
            K = pairwise_kernels(X,self.X_,metric=self.metric, degree=self.degree, coef0=self.coef0, gamma=self.gamma)
        elif self.metric == 'sigmoid':
            K = pairwise_kernels(X,self.X_,metric=self.metric, coef0=self.coef0, gamma=self.gamma)
        else:
            raise ValueError('metric not supported')
        
        return np.mean(np.abs( (y_test-y).T @ K @ self.y_)/self.normalizing_constant)


