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
    #loading csvs
    x=pd.read_csv('../resources/solar_system_census.csv',index_col=0).to_numpy()
    y=pd.read_csv('../resources/solar_system_census_planets.csv',index_col=0).to_numpy()
    categories=["The flying cities of Venus","United Nations of Earth","Mars Republic","The Asteroids' Belt colonies"]

    parser = argparse.ArgumentParser(description="A logistic regression classifier that can discriminate between two classes.")
    parser.add_argument('-zipcode',type=int,choices=range(4),required=True)
    args=parser.parse_args()
    zipcode=args.zipcode
    (x_train,x_test,y_train,y_test)=data_spliter(x,y,.8)

    #creating 0/1 scaled dataset
    new_y_train=(y_train==zipcode).astype(float)
    new_y_test=(y_test==zipcode).astype(float)
    myLR=MLR(np.ones((x.shape[1]+1,1)),max_iter=2*10**5)
    myLR.fit_(x_train,new_y_train)
    train_pred=myLR.predict_(x_train)
    print(f'Train set loss: {myLR.loss_(new_y_train,train_pred)}.')
    test_pred=myLR.predict_(x_test)
    print(f'Test set loss: {myLR.loss_(new_y_test,test_pred)}.')

    train_pred[train_pred>=.5]=1
    train_pred[train_pred<.5]=0
    test_pred[test_pred>=.5]=1
    test_pred[test_pred<.5]=0

    correct_train=train_pred==new_y_train
    train_accuracy=correct_train.mean()
    print(f'Train set accuracy: {train_accuracy}.')

    correct_test=test_pred==new_y_test
    test_accuracy=correct_test.mean()
    print(f'Test set accuracy: {test_accuracy}.')

    all_x=np.vstack([x_train,x_test])
    all_y=np.vstack([new_y_train,new_y_test])
    all_pred=np.vstack([train_pred,test_pred])
    all_correct=np.vstack([correct_train,correct_test])

    #ploting values
    planet=categories[zipcode]

    hue_vector=np.where(all_y,planet,'Other zipcodes').ravel()
    style_vector=np.where(all_correct,'Correct Label','Incorrect Label').ravel()
    sns.set(style='darkgrid')
    sns.scatterplot(x=all_x[:,0],y=all_x[:,1],hue=hue_vector,hue_order=[planet,'Other zipcodes'],style=style_vector,markers={'Correct Label':'o','Incorrect Label':'X'})
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.show()

    hue_vector=np.where(all_y,planet,'Other zipcodes').ravel()
    style_vector=np.where(all_correct,'Correct Label','Incorrect Label').ravel()
    sns.scatterplot(x=all_x[:,0],y=all_x[:,2],hue=hue_vector,hue_order=[planet,'Other zipcodes'],style=style_vector,markers={'Correct Label':'o','Incorrect Label':'X'})
    plt.xlabel('Height')
    plt.ylabel('Bone density')
    plt.show()

    hue_vector=np.where(all_y,planet,'Other zipcodes').ravel()
    style_vector=np.where(all_correct,'Correct Label','Incorrect Label').ravel()
    sns.scatterplot(x=all_x[:,1],y=all_x[:,2],hue=hue_vector,hue_order=[planet,'Other zipcodes'],style=style_vector,markers={'Correct Label':'o','Incorrect Label':'X'})
    plt.xlabel('Weight')
    plt.ylabel('Bone density')
    plt.show()

    #ploting 3d scatter plot
    fig=plt.figure()
    ax=plt.axes(projection='3d')
    correct_predictions=(all_correct==1).ravel()
    scatter1=ax.scatter(all_x[correct_predictions,0],all_x[correct_predictions,1],all_x[correct_predictions,2],c='green',label='Correct', marker='o')
    incorrect_predictions=(all_correct==0).ravel()
    colors=all_y[incorrect_predictions,:].astype(int)
    scatter2=ax.scatter(all_x[incorrect_predictions, 0],all_x[incorrect_predictions, 1],all_x[incorrect_predictions, 2],c='red',label='Incorrect',marker='X')
    ax.set_xlabel('Height')
    ax.set_ylabel('Weight')
    ax.set_zlabel('Bone Density')
    ax.legend()
    plt.show()



