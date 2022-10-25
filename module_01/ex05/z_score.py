import numpy as np

def zscore(x):
    if not isinstance(x,np.ndarray) or not x.size or x.ndim!=2 or not np.issubdtype(x.dtype,np.number):
        print('x: has to be an numpy.ndarray, a vector.')
        return None
    mu=x.mean(axis=0)
    sigma=x.std(axis=0)
    return (x-mu)/sigma