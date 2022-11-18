import numpy as np

def log_loss_(y,y_hat,eps=1e-15):
    if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
        print('y has to be an numpy array, vector of dim (m,1)')
        return None
    if not isinstance(y_hat,np.ndarray)or not y_hat.size or y_hat.ndim>2 or y_hat.shape[1]!=1 or not np.issubdtype(y_hat.dtype,np.number):
        print('y_hat has to be an numpy array, vector of dim (m,1)')
        return None
    if y.shape != y_hat.shape:
        print('y and y_hat has different shapes.')
        return None
    if not isinstance(eps,float):
        print('eps has to be a float.')
        return None
    log_error=y*np.log(y_hat+eps)+(1-y)*np.log(1-y_hat+eps)
    return float(-log_error.sum()/y.shape[0])
