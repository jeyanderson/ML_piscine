import numpy as np
import sys
sys.path.append('../')
from ex01.l2_reg import l2

def reg_log_loss_(y,y_hat,theta,lambda_,eps=1e-15):
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
    return float(-(y.T@np.log(y_hat+eps)+(1-y).T@np.log(1-y_hat+eps))/y.shape[0]+lambda_*l2(theta)/(2*y.shape[0]))