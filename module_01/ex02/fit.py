import numpy as np

def gradient(x,y,theta):
    X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
    return X.T@(X@theta-y)/X.shape[0]

def fit_(x,y,theta,alpha,max_iter):
    if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
        print('y has to be a numpy array, vector of dim (x,1)')
        return None
    if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or x.shape[1]!=1 or not np.issubdtype(x.dtype,np.number):
        print('x has to be a numpy array, vector of dim (x,1)')
        return None
    if not isinstance(theta,np.ndarray)or not theta.size or theta.ndim>2 or theta.shape!=(2,1) or not np.issubdtype(x.dtype,np.number):
        print('theta has to be a numpy array, vector of dim (2,1)')
        return None
    if not isinstance(alpha,float) or alpha<=0:
        print('alpha has to be a strictrly positive float.')
        return None
    if not isinstance(max_iter,int) or max_iter<0:
        print('max_iter has to be an equal or bigger than 0 int.')
        return None
    if y.shape != x.shape:
        print('y and x has different shapes.')
        return None
    for _ in range(max_iter):
        grad=gradient(x,y,theta)
        theta=theta-alpha*grad
    return theta