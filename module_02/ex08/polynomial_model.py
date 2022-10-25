import numpy as np

def add_polynomial_features(x,power):
    if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or not np.issubdtype(x.dtype,np.number):
        print('x has to be a numpy array, vector of dim (x,1)')
        return None
    if not isinstance(power,int):
        print('power has to be an int.')
        return None
    poly=x
    for i in range(2,power+1):
        poly=np.hstack((poly,x**i))
    return poly