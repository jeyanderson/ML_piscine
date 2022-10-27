import numpy as np

class MyLinearRegression():
    def __init__(self,thetas,alpha=0.001,max_iter=1000):
        err_msg='thetas has to be either a list or a np array, a vector of shape (2,1).'
        if isinstance(thetas,np.ndarray):
            if not isinstance(thetas,np.ndarray)or not thetas.size or thetas.ndim>2 or not np.issubdtype(thetas.dtype,np.number):
                print(err_msg)
                return None
        elif isinstance(thetas,list):
            try:
                thetas=np.array(thetas).reshape((-1,1))
                assert np.issubdtype(thetas.dtype,np.number)
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
        self.thetas=thetas
        self.alpha=alpha
        self.max_iter=max_iter
    def fit_(self,x,y):
        if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or not np.issubdtype(x.dtype,np.number):
            print('x has to be a numpy array, vector of dim (x,1)')
            return None
        if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
            print('y has to be a numpy array, vector of dim (x,1)')
            return None
        if y.shape[0] != x.shape[0]:
            print('y and x has different shapes.')
            return None
        X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        for _ in range(self.max_iter):
            grad=X.T@(X@self.thetas-y)/X.shape[0]
            self.thetas=self.thetas-self.alpha*grad
    def predict_(self,x):
        if not isinstance(x,np.ndarray)or not x.size or x.ndim!=2 or not np.issubdtype(x.dtype,np.number):
            print('x has to be an numpy.array, a matrix of dimension m*n.')
            return None
        X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        return X@self.thetas
    def loss_elem_(self,y,y_hat):
        if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
            print('y has to be an numpy array, vector of dim (x,1)')
            return None
        if not isinstance(y_hat,np.ndarray)or not y_hat.size or y_hat.ndim>2 or y_hat.shape[1]!=1 or not np.issubdtype(y_hat.dtype,np.number):
            print('y_hat has to be an numpy array, vector of dim (x,1)')
            return None
        if y.shape != y_hat.shape:
            print('y and y_hat has different shapes.')
            return None
        return (y-y_hat)**2
    def loss_(self,y,y_hat):
        squared_loss=self.loss_elem_(y,y_hat)
        if squared_loss is None:
            return None
        return squared_loss.sum()/(2*y.shape[0])
    def mse_(self,y,y_hat):
        if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
            print('y has to be an numpy array, vector of dim (x,1)')
            return None
        if not isinstance(y_hat,np.ndarray)or not y_hat.size or y_hat.ndim>2 or y_hat.shape[1]!=1 or not np.issubdtype(y_hat.dtype,np.number):
            print('y_hat has to be an numpy array, vector of dim (x,1)')
            return None
        if y.shape != y_hat.shape:
            print('y and y_hat has different shapes.')
            return None
        error=y-y_hat
        return float(error.T.dot(error)/y.shape[0])

class MyRidge(MyLinearRegression):
    def __init__(self,thetas,alpha=0.001,max_iter=1000,lambda_=.5):
        super().__init__(thetas,alpha,max_iter)
        if not isinstance(lambda_,(int,float)) or lambda_<0:
            print('lambda has to be a positive float.')
            return None
        self.lambda_=lambda_
    def get_params_(self):
        if not hasattr(self,'thetas'):
            return None
        return self.thetas
    def set_params_(self,thetas):
        err_msg='thetas has to be either a list or a np array, a vector.'
        if isinstance(thetas,np.ndarray):
            if not isinstance(thetas,np.ndarray)or not thetas.size or thetas.ndim>2 or not np.issubdtype(thetas.dtype,np.number):
                print(err_msg)
                return None
        elif isinstance(thetas,list):
            try:
                thetas=np.array(thetas).reshape((-1,1))
                assert np.issubdtype(thetas.dtype,np.number)
            except Exception:
                print(err_msg)
                return None
        else:
            print(err_msg)
            return None
        self.thetas=thetas
    def loss_(self,y,y_hat):
        if not hasattr(self,'thetas') or not hasattr(self,'lambda_'):
            return None
        squared_loss=self.loss_elem_(y,y_hat)
        if squared_loss is None:
            return None
        reg=self.thetas[1:,:].T.dot(self.thetas[1:,:]).item()
        return squared_loss.sum()*self.lambda_*reg/(2*y.shape[0])
    def gradient_(self,x,y):
        if not hasattr(self,'thetas') or not hasattr(self,'lambda_'):
            return None
        if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
            print('y has to be an numpy array, vector of dim (x,1)')
            return None
        if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or not np.issubdtype(x.dtype,np.number):
            print('y_hat has to be an numpy array, vector of dim (x,1)')
            return None
        if y.shape[0]!=x.shape[0]:
            print('y and y_hat has different shapes.')
            return None
        if self.thetas.shape[0]!=x.shape[1]+1:
            print('thetas and x shapes doesnt match.')
            return None
        m=x.shape[0]
        X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        pred=X@self.thetas
        theta_prime=self.thetas.copy()
        theta_prime[0,0]=0
        return (X.T@(pred-y)+self.lambda_*theta_prime)/m
    def fit_(self,x,y):
        if not hasattr(self,'thetas') or not hasattr(self,'lambda_'):
            return None
        if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
            print('y has to be an numpy array, vector of dim (x,1)')
            return None
        if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or not np.issubdtype(x.dtype,np.number):
            print('y_hat has to be an numpy array, vector of dim (x,1)')
            return None
        if y.shape[0]!=x.shape[0]:
            print('y and y_hat has different shapes.')
            return None
        if self.thetas.shape[0]!=x.shape[1]+1:
            print('thetas and x shapes doesnt match.')
            return None
        m=x.shape[0]
        X=np.concatenate((np.ones((x.shape[0],1)),x),axis=1)
        for _ in range(self.max_iter):
            theta_prime=self.thetas.copy()
            theta_prime[0,0]=0
            grad=(X.T@(X@self.thetas-y)+self.lambda_*theta_prime)/m
            self.thetas=self.thetas-self.alpha*grad