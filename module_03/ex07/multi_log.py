import numpy as np
import argparse
import sys 
sys.path.append('../')
import pandas as pd
from numpy.random import default_rng
from ex06.my_logistic_regression import MyLogisticRegression as MLR
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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
    x=pd.read_csv('../resources/solar_system_census.csv',index_col=0).to_numpy()
    y=pd.read_csv('../resources/solar_system_census_planets.csv',index_col=0).to_numpy()
    categories=["The flying cities of Venus","United Nations of Earth","Mars Republic","The Asteroids' Belt colonies"]
    (x_train,x_test,y_train,y_test)=data_spliter(x,y,.8)

    zipcode=0
    new_y_train=(y_train==zipcode).astype(float)
    new_y_test=(y_test==zipcode).astype(float)
    myLR=MLR(np.ones((x.shape[1]+1,1)),max_iter=2*10**5)
    myLR.fit_(x_train,new_y_train)
    train_pred1=myLR.predict_(x_train)
    print(f'Train set loss: {myLR.loss_(new_y_train,train_pred1)}.')
    test_pred1=myLR.predict_(x_test)
    print(f'Test set loss: {myLR.loss_(new_y_test,test_pred1)}.')

    zipcode=2
    new_y_train=(y_train==zipcode).astype(float)
    new_y_test=(y_test==zipcode).astype(float)
    myLR=MLR(np.ones((x.shape[1]+1,1)),max_iter=2*10**5)
    myLR.fit_(x_train,new_y_train)
    train_pred2=myLR.predict_(x_train)
    print(f'Train set loss: {myLR.loss_(new_y_train,train_pred2)}.')
    test_pred2=myLR.predict_(x_test)
    print(f'Test set loss: {myLR.loss_(new_y_test,test_pred2)}.')

    zipcode=3
    new_y_train=(y_train==zipcode).astype(float)
    new_y_test=(y_test==zipcode).astype(float)
    myLR=MLR(np.ones((x.shape[1]+1,1)),max_iter=2*10**5)
    myLR.fit_(x_train,new_y_train)
    train_pred3=myLR.predict_(x_train)
    print(f'Train set loss: {myLR.loss_(new_y_train,train_pred3)}.')
    test_pred3=myLR.predict_(x_test)
    print(f'Test set loss: {myLR.loss_(new_y_test,test_pred3)}.')

    zipcode=4
    new_y_train=(y_train==zipcode).astype(float)
    new_y_test=(y_test==zipcode).astype(float)
    myLR=MLR(np.ones((x.shape[1]+1,1)),max_iter=2*10**5)
    myLR.fit_(x_train,new_y_train)
    train_pred4=myLR.predict_(x_train)
    print(f'Train set loss: {myLR.loss_(new_y_train,train_pred4)}.')
    test_pred4=myLR.predict_(x_test)
    print(f'Test set loss: {myLR.loss_(new_y_test,test_pred4)}.')

    training_pred=np.hstack([train_pred1,train_pred2,train_pred3,train_pred4])
    training_pred=np.argmax(training_pred,axis=1).reshape(-1,1)
    testing_pred=np.hstack([test_pred1,test_pred2,test_pred3,test_pred4])
    testing_pred=np.argmax(testing_pred,axis=1).reshape(-1,1)

    correct_train=training_pred==y_train
    train_accuracy=correct_train.mean()
    print(f'Train accuracy: {train_accuracy}.')

    correct_test=testing_pred==y_test
    test_accuracy=correct_test.mean()
    print(f'Test accuracy: {test_accuracy}.')

    all_x=np.vstack([x_test,x_train])
    all_y=np.vstack([y_test,y_train])
    all_predictions=np.vstack([testing_pred,training_pred])
    all_correct=np.vstack([correct_train,correct_test])

    sns.set(style='darkgrid')

    sns.scatterplot(x=all_x[:,0],y=all_x[:,1],hue=all_y.astype(int).ravel(),style=all_correct.ravel(),markers={1:'o', 0:'X'},palette='deep')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.show()
    sns.scatterplot(x=all_x[:,0],y=all_x[:,2],hue=all_y.astype(int).ravel(),style=all_correct.ravel(),markers={1:'o', 0:'X'},palette='deep')
    plt.xlabel('Height')
    plt.ylabel('Bone density')
    plt.show()
    sns.scatterplot(x=all_x[:,0],y=all_x[:,1],hue=all_y.astype(int).ravel(),style=all_correct.ravel(),markers={1:'o', 0:'X'},palette='deep')
    plt.xlabel('Weight')
    plt.ylabel('Bone density')
    plt.show()