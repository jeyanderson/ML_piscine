import numpy as np

def gradient(x,y,theta):
    if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or not np.issubdtype(x.dtype,np.number):
        print('x has to be a numpy array, vector of dim (x,1)')
        return None
    if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
        print('y has to be a numpy array, vector of dim (x,1)')
        return None
    if not isinstance(theta,np.ndarray)or not theta.size or theta.ndim>2 or not np.issubdtype(x.dtype,np.number):
        print('theta has to be a numpy array, vector of dim (2,1)')
        return None
    if y.shape[0] != x.shape[0]:
        print('y and x has different shapes.')
        return None
    X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
    return X.T@(X@theta-y)/X.shape[0]