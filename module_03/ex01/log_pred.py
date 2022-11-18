import numpy as np
import sys
sys.path.append('../') 
from ex00.sigmoid import sigmoid_

def logistic_predict_(x,theta):
    if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or not np.issubdtype(x.dtype,np.number):
        print('x has to be a numpy array, vector of dim (m,n)')
        return None
    if not isinstance(theta,np.ndarray)or not theta.size or theta.ndim>2 or not np.issubdtype(theta.dtype,np.number):
        print('theta has to be a np array, a vector of shape (n+1,1).')
        return None
    X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
    return sigmoid_(X@theta)
    