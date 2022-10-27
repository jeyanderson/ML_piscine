import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from ex00.polynomial_model_extended import add_polynomial_features
from ex06.ridge import MyRidge as MR
from numpy.random import default_rng

def data_spliter(x,y,proportion):
    if not isinstance(x,np.ndarray)or not x.size or x.ndim>2 or not np.issubdtype(x.dtype,np.number):
            print('x has to be a numpy array, vector of dim (x,1)')
            return None
    if not isinstance(y,np.ndarray)or not y.size or y.ndim>2 or y.shape[1]!=1 or not np.issubdtype(y.dtype,np.number):
        print('y has to be a numpy array, vector of dim (x,1)')
        return None
    if y.shape[0] != x.shape[0]:
        print('y and x has different shapes.')
        return None
    if not isinstance(proportion,float) or proportion<0 or proportion>1:
        print('proportion has to be a float >=1 & <=99.')
        return None
    rng=default_rng(1444)
    z=np.hstack((x,y))
    rng.shuffle(z)
    x,y=z[:,:-1].reshape(x.shape),z[:,-1].reshape(y.shape)
    idx=int(x.shape[0]*proportion)
    x_train,x_test=np.split(x,[idx])
    y_train,y_test=np.split(y,[idx])
    return(x_train,x_test,y_train,y_test)

if __name__=='__main__':
    df=pd.read_csv('../resources/space_avocado.csv',index_col=0)
    x=np.array(df[['weight','prod_distance','time_delivery']])
    n_features=x.shape[1]
    y=np.array(df['target']).reshape(-1,1)
    (x_rest,x_test,y_rest,y_test)=data_spliter(x,y,.8)
    (x_train,x_valid,y_train,y_valid)=data_spliter(x_rest,y_rest,.8)

    degrees=range(1,5)
    lambdas=np.arange(0,1,.2).tolist()
    models=[]

    for degree in degrees:
        mse_list=[]
        models_list=[]
        x_train_poly=add_polynomial_features(x_train,degree)
        x_valid_poly=add_polynomial_features(x_valid,degree)
        x_test_poly=add_polynomial_features(x_test,degree)

        min=x_train_poly.min(axis=0)
        range=x_train_poly.max(axis=0)-min
        x_train_poly=(x_train_poly-min)/range
        x_valid_poly=(x_valid_poly-min)/range
        x_test_poly=(x_test_poly-min)/range

        for lambda_ in lambdas:
            mr=MR(np.ones((1+n_features*degree,1)),max_iter=10**5,alpha=1e-1,lambda_=lambda_)
            mr.fit_(x_train_poly,y_train)
            pred=mr.predict_(x_train_poly)
            train_mse=mr.mse_(y_train,pred)
            print(f'Train set mse: {train_mse}.')

            pred=mr.predict_(x_valid_poly)
            valid_mse=mr.mse_(y_valid,pred)
            print(f'Valid set mse: {valid_mse}.')

            mse_list.append(valid_mse)
            models_list.append(mr)

            thetas=mr.get_params_().copy()
            thetas.resize((16,1))
            models.append(thetas)
        
        mse_list=np.array(mse_list)
        idx=np.argmin(mse_list)
        best_lambda=lambdas[idx]
        best_model=models_list[idx]
        print(f'The best value of lambda for the model with polynomial degree {degree} is {best_lambda}.')
        
        pred=best_model.predict_(x_test_poly)
        test_mse=best_model.mse_(y_test,pred)
        print(f'best model mse: {test_mse}.')

    all_models=np.hstack(models)
    pd.DataFrame(all_models).to_csv('models.csv',float_format='%010.1f',index=False)