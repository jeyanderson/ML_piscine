import numpy as np
import sys
sys.path.append('../') 
from ex00.sigmoid import sigmoid_

def log_gradient(x,y,theta):
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
    gradient=np.zeros(theta.shape)
    X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
    m=x.shape[0]
    n=x.shape[1]
    pred=sigmoid_(X@theta)
    gradient[0,0]=(pred-y).sum()/m
    for j in range(1,n+1):
        gradient[j,0]=(pred-y).T.dot(x[:,j-1]).sum()/m
    return gradient