import numpy as np

def predict_(x,theta):
    if not isinstance(x,np.ndarray)or not x.size or x.ndim!=2 or not np.issubdtype(x.dtype,np.number):
        print('x has to be an numpy.array, a matrix of dimension m*n.')
        return None
    if not isinstance(theta,np.ndarray)or not theta.size or theta.shape!=(x.shape[1]+1,1)or not np.issubdtype(theta.dtype,np.number):
        print('theta has to be an numpy.array, a vector of dimension (n+1)*1')
        return None
    X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
    return X@theta