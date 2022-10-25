import numpy as np
from numpy.random import default_rng

def data_spliter(x,y,proportion):
    if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or not np.issubdtype(x.dtype,np.number):
            print('x has to be a numpy array, vector of dim (x,1)')
            return None
    if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
        print('y has to be a numpy array, vector of dim (x,1)')
        return None
    if y.shape[0] != x.shape[0]:
        print('y and x has different shapes.')
        return None
    if not isinstance(proportion,float) or proportion<0 or proportion>1:
        print('proportion has to be a float >=1 & <=99.')
        return None
    rng=default_rng(1444)
    z=np.hstack((x,y))
    rng.shuffle(z)
    x,y=z[:,:-1].reshape(x.shape),z[:,-1].reshape(y.shape)
    idx=int(x.shape[0]*proportion)
    x_train,x_test=np.split(x,[idx])
    y_train,y_test=np.split(y,[idx])
    return(x_train,x_test,y_train,y_test)

    