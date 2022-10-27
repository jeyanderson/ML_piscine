import numpy as np

def iterative_l2(theta):
    if not isinstance(theta,np.ndarray)or not theta.size or theta.ndim>2 or not np.issubdtype(theta.dtype,np.number):
        print('theta has to be a numpy array, vector of dim (m,1)')
        return None
    l2=0
    for i in range(1,theta.shape[0]):
        l2+=theta[i,0]**2
    return l2

def l2(theta):
    if not isinstance(theta,np.ndarray)or not theta.size or theta.ndim>2 or not np.issubdtype(theta.dtype,np.number):
        print('theta has to be a numpy array, vector of dim (m,1)')
        return None
    theta_prime=theta.copy()
    theta_prime[0,0]=0
    return (theta_prime.T@theta_prime).item()