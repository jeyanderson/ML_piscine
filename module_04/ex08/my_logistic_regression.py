import numpy as np

class MyLogisticRegression:
    def __init__(self,theta,alpha=.001,max_iter=1000,penalty='l2',lambda_=.5):
        err_msg='theta has to be either a list or a np array, a vector of shape (2,1).'
        if isinstance(theta,np.ndarray):
            if not isinstance(theta,np.ndarray)or not theta.size or theta.ndim>2 or not np.issubdtype(theta.dtype,np.number):
                print(err_msg)
                return None
        elif isinstance(theta,list):
            try:
                theta=np.array(theta).reshape((-1,1))
                assert np.issubdtype(theta.dtype,np.number)
            except Exception:
                print(err_msg)
                return None
        else:
            print(err_msg)
            return None
        if not isinstance(alpha,float) or alpha<=0:
            print('alpha has to be a strictrly positive float.')
            return None
        if not isinstance(max_iter,int) or max_iter<0:
            print('max_iter has to be an equal or bigger than 0 int.')
            return None
        supported_penalities=['l2']
        self.theta=theta
        self.alpha=alpha
        self.max_iter=max_iter
        self.lambda_=lambda_ if penalty in supported_penalities else 0
    def sigmoid_(self,x):
        if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or not np.issubdtype(x.dtype,np.number):
            print('x has to be a numpy array, vector of dim (x,1)')
            return None
        return 1/(1+np.exp(-x))
    def fit_(self,x,y):
        if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or not np.issubdtype(x.dtype,np.number):
            print('x has to be a numpy array, vector of dim (m,n)')
            return None
        if not isinstance(y,np.ndarray)or not y.size or x.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
            print('y has to be a numpy array, vector of dim (m,1)')
            return None
        if not isinstance(self.theta,np.ndarray)or not self.theta.size or self.theta.ndim>2 or self.theta.shape[1]!=1 or not np.issubdtype(self.theta.dtype,np.number):
            print('theta has to be a numpy array, vector of dim (n+1,1)')
            return None
        if x.shape[0]!=y.shape[0]:
            print('x and y must have the same number of rows.')
            return None
        X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        if self.lambda_==0:
            for _ in range(self.max_iter):
                pred=self.sigmoid_(X@self.theta)
                grad=X.T@(pred-y)/y.shape[0]
                self.theta=self.theta-self.alpha*grad
        else:
            for _ in range(self.max_iter):
                pred=self.sigmoid_(X@self.theta)
                theta_prime=self.theta.copy()
                theta_prime[0,0]=0
                grad=(X.T@(pred-y)+self.lambda_*theta_prime)/y.shape[0]
                self.theta=self.theta-self.alpha*grad

    def predict_(self,x):
        if not hasattr(self,'theta')or not hasattr(self,'alpha')or not hasattr(self,'max_iter'):
            return None
        if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or not np.issubdtype(x.dtype,np.number):
            print('x has to be a numpy array, vector of dim (m,n)')
            return None
        X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        return self.sigmoid_(X@self.theta)
    def loss_elem_(self,y,y_hat,eps=1e-15):
        if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
            print('y has to be an numpy array, vector of dim (x,1)')
            return None
        if not isinstance(y_hat,np.ndarray)or not y_hat.size or y_hat.ndim>2 or y_hat.shape[1]!=1 or not np.issubdtype(y_hat.dtype,np.number):
            print('y_hat has to be an numpy array, vector of dim (x,1)')
            return None
        if y.shape != y_hat.shape:
            print('y and y_hat has different shapes.')
            return None
        if not isinstance(eps,float):
            print('eps has to be a float.')
            return None
        return y*np.log(y_hat+eps)+(1-y)*np.log(1-y_hat+eps)
    def loss_(self,y,y_hat):
        log_error=self.loss_elem_(y,y_hat)
        if log_error is None:
            return None
        reg=self.theta[1:,:].T.dot(self.theta[1:,:]).item()
        return float(-log_error.sum()/y.shape[0]+self.lambda_*reg/(2*y.shape[0]))