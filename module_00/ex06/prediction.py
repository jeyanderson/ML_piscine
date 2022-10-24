import numpy as np
def predict_(x,theta):
    if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or x.shape[1]!=1 or not np.issubdtype(x.dtype,np.number):
        print('x has to be a numpy array, vector of dim (x,1)')
        return None
    if not isinstance(theta,np.ndarray)or not theta.size or theta.ndim>2 or theta.shape!=(2,1) or not np.issubdtype(x.dtype,np.number):
        print('theta has to be a numpy array, vector of dim (2,1)')
        return None
    x=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
    return x @ theta