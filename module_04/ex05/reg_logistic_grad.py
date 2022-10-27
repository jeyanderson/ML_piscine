import numpy as np

def sigmoid_(x):
    if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or not np.issubdtype(x.dtype,np.number):
            print('x has to be a numpy array, vector of dim (x,1)')
            return None
    return 1/(1+np.exp(-x))

def reg_logistic_grad(y,x,theta,lambda_):
    if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
        print('y has to be an numpy array, vector of dim (x,1)')
        return None
    if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or not np.issubdtype(x.dtype,np.number):
        print('y_hat has to be an numpy array, vector of dim (x,1)')
        return None
    if y.shape[0] != x.shape[0]:
        print('y and y_hat has different shapes.')
        return None
    if not isinstance(theta,np.ndarray)or not theta.size or theta.ndim>2 or not np.issubdtype(theta.dtype,np.number):
        print('theta has to be a numpy array, vector of dim (m,1)')
        return None
    if not isinstance(lambda_,(int,float)):
        print('labmda has to be a float.')
        return None
    gradient=np.zeros(theta.shape)
    m=x.shape[0]
    n=x.shape[1]
    X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
    pred=sigmoid_(X@theta)
    gradient[0,0]=(pred-y).sum()/m
    for j in range(1,n+1):
        gradient[j,0]=((pred-y).T.dot(x[:,j-1])+lambda_*theta[j,0])/m
    return gradient

def vec_reg_logistic_grad(y,x,theta,lambda_):
    if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
        print('y has to be an numpy array, vector of dim (x,1)')
        return None
    if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or not np.issubdtype(x.dtype,np.number):
        print('y_hat has to be an numpy array, vector of dim (x,1)')
        return None
    if y.shape[0] != x.shape[0]:
        print('y and y_hat has different shapes.')
        return None
    if not isinstance(theta,np.ndarray)or not theta.size or theta.ndim>2 or not np.issubdtype(theta.dtype,np.number):
        print('theta has to be a numpy array, vector of dim (m,1)')
        return None
    if not isinstance(lambda_,(int,float)):
        print('labmda has to be a float.')
        return None
    m=x.shape[0]
    X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
    pred=sigmoid_(X@theta)
    theta_prime=theta.copy()
    theta_prime[0,0]=0
    return (X.T@(pred-y)+lambda_*theta_prime)/m