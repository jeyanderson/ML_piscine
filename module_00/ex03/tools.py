from ast import Add
import numpy as np
def add_intercept(x):
    if not isinstance(x,np.ndarray) or not x.size or not np.issubdtype(x.dtype,np.number):
        return None
    return np.concatenate((np.ones((x.shape[0],1)),x),axis=1)