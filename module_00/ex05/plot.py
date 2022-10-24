import numpy as np
import matplotlib.pyplot as plt

def plot(x,y,theta):
    if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or x.shape[1]!=1 or not np.issubdtype(x.dtype,np.number):
        print('x has to be a numpy array, vector of dim (x,1)')
        return None
    if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
        print('y has to be a numpy array, vector of dim (x,1)')
        return None
    if not isinstance(theta,np.ndarray)or not theta.size or theta.ndim>2 or theta.shape!=(2,1) or not np.issubdtype(x.dtype,np.number):
        print('theta has to be a numpy array, vector of dim (2,1)')
        return None
    plt.scatter(x,y)
    plt.plot(x,x*theta[1,0]+theta[0,0],color='red')
    plt.show()

x = np.arange(1,6).reshape(-1,1)
y = np.array([[3.74013816],[3.61473236],[4.57655287],[4.66793434],[5.95585554]])
theta1=np.array([[4.5],[-0.2]])
plot(x,y,theta1)