import numpy as np

def minmax(x):
    if not isinstance(x,np.ndarray) or not x.size or x.ndim!=2 or not np.issubdtype(x.dtype,np.number):
        print('x: has to be an numpy.ndarray, a vector.')
        return None
    min=x.min(axis=0)
    max=x.max(axis=0)
    return (x-min)/(max-min)