import numpy as np

def loss_elem_(y,y_hat):
    if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
        print('y has to be a numpy array, vector of dim (x,1)')
        return None
    if not isinstance(y_hat,np.ndarray)or not y_hat.size or y_hat.ndim>2 or y_hat.shape[1]!=1 or not np.issubdtype(y_hat.dtype,np.number):
        print('y_hat has to be a numpy array, vector of dim (x,1)')
        return None
    if y.shape != y_hat.shape:
        print('y and y_hat has different shapes.')
        return None
    return (y-y_hat)**2
def loss_(y,y_hat):
    squared_loss=loss_elem_(y,y_hat)
    if squared_loss is None:
        return None
    return squared_loss/(2*y.shape[0])