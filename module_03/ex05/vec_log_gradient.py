import numpy as np
import sys
sys.path.append('../')
from ex00.sigmoid import sigmoid_

def vec_log_gradient(x,y,theta):
    if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or not np.issubdtype(x.dtype,np.number):
        print('x has to be a numpy array, vector of dim (m,n)')
        return None
    if not isinstance(y,np.ndarray)or not y.size or x.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
        print('y has to be a numpy array, vector of dim (m,1)')
        return None
    if not isinstance(theta,np.ndarray)or not theta.size or theta.ndim>2 or theta.shape[1]!=1 or not np.issubdtype(theta.dtype,np.number):
        print('theta has to be a numpy array, vector of dim (n+1,1)')
        return None
    if x.shape[0]!=y.shape[0]:
        print('x and y must have the same number of rows.')
        return None
    m=x.shape[0]
    X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
    pred=sigmoid_(X@theta)
    return X.T@(pred-y)/m