import numpy as np
import sys
sys.path.append('../')
from ex01.l2_reg import l2

def reg_loss_(y,y_hat,theta,lambda_):
        if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
            print('y has to be an numpy array, vector of dim (x,1)')
            return None
        if not isinstance(y_hat,np.ndarray)or not y_hat.size or y_hat.ndim>2 or y_hat.shape[1]!=1 or not np.issubdtype(y_hat.dtype,np.number):
            print('y_hat has to be an numpy array, vector of dim (x,1)')
            return None
        if y.shape != y_hat.shape:
            print('y and y_hat has different shapes.')
            return None
        if not isinstance(theta,np.ndarray)or not theta.size or theta.ndim>2 or not np.issubdtype(theta.dtype,np.number):
            print('theta has to be a numpy array, vector of dim (m,1)')
            return None
        if not isinstance(lambda_,(int,float)):
            print('labmda has to be a float.')
            return None
        error=y_hat-y
        return float((error.T.dot(error)+lambda_*l2(theta))/(2*y.shape[0]))