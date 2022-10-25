import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MLR
from data_spliter import data_spliter

#finding done by changing alpha and max_iter

df=pd.read_csv('../resources/space_avocado.csv',index_col=0)
x=np.array(df[['weight','prod_distance','time_delivery']])
y=np.array(df['target']).reshape(-1,1)
x_train,x_test,y_train,y_test=data_spliter(x,y,.8)

train_mse_list=[]
test_mse_list=[]

print('testing polymonial degree 1.')
myLR1=MLR(np.ones((x.shape[1]+1,1)).reshape(-1,1),alpha=1e-1,max_iter=10000000)
min=x_train.min(axis=0)
range=x_train.max(axis=0)-min
normalized_x_train=(x_train-min)/range
myLR1.fit_(normalized_x_train,y_train)
pred=myLR1.predict_(normalized_x_train)
mse=myLR1.mse_(y_train,pred)
print(f'training mse on polynomial degree 1: {mse}.')
train_mse_list.append(mse)

normalized_x_test=(x_test-min)/range
test_pred=myLR1.predict_(normalized_x_test)
mse=myLR1.mse_(y_test,test_pred)
print(f'testing mse on polynomial degree 1: {mse}.')
test_mse_list.append(mse)

print('testing polymonial degree 2.')
poly_x_train = np.hstack([add_polynomial_features(row.reshape(-1, 1), 2)for row in x_train.T])
min=poly_x_train.min(axis=0)
range=poly_x_train.max(axis=0)-min
normalized_poly_x_train=(poly_x_train-min)/range
myLR2=MLR(np.ones((normalized_poly_x_train.shape[1]+1,1)),alpha=1e-1,max_iter=10000000)
myLR2.fit_(normalized_poly_x_train,y_train)
pred=myLR2.predict_(normalized_poly_x_train)
mse=myLR2.mse_(y_train,pred)
print(f'training mse on polynomial degree 2: {mse}.')
train_mse_list.append(mse)

poly_x_test = np.hstack([add_polynomial_features(row.reshape(-1, 1), 2)for row in x_test.T])
normalized_x_test=(poly_x_test-min)/range
test_pred=myLR2.predict_(normalized_x_test)
mse=myLR2.mse_(y_test,test_pred)
print(f'testing mse on polynomial degree 2: {mse}.')
test_mse_list.append(mse)

print('testing polymonial degree 3.')
poly_x_train = np.hstack([add_polynomial_features(row.reshape(-1,1),3)for row in x_train.T])
min=poly_x_train.min(axis=0)
range=poly_x_train.max(axis=0)-min
normalized_poly_x_train=(poly_x_train-min)/range
myLR3=MLR(np.ones((normalized_poly_x_train.shape[1]+1,1)),alpha=1e-1,max_iter=10000000)
myLR3.fit_(normalized_poly_x_train,y_train)
pred=myLR3.predict_(normalized_poly_x_train)
mse=myLR3.mse_(y_train,pred)
print(f'training mse on polynomial degree 3: {mse}.')
train_mse_list.append(mse)

poly_x_test = np.hstack([add_polynomial_features(row.reshape(-1,1),3)for row in x_test.T])
normalized_x_test=(poly_x_test-min)/range
test_pred=myLR3.predict_(normalized_x_test)
mse=myLR3.mse_(y_test,test_pred)
print(f'testing mse on polynomial degree 3: {mse}.')
test_mse_list.append(mse)

print('testing polymonial degree 4.')
poly_x_train = np.hstack([add_polynomial_features(row.reshape(-1,1),4)for row in x_train.T])
min=poly_x_train.min(axis=0)
range=poly_x_train.max(axis=0)-min
normalized_poly_x_train=(poly_x_train-min)/range
myLR4=MLR(np.ones((normalized_poly_x_train.shape[1]+1,1)),alpha=1e-1,max_iter=10000000)
myLR4.fit_(normalized_poly_x_train,y_train)
pred=myLR4.predict_(normalized_poly_x_train)
mse=myLR4.mse_(y_train,pred)
print(f'training mse on polynomial degree 4: {mse}.')
train_mse_list.append(mse)

poly_x_test = np.hstack([add_polynomial_features(row.reshape(-1,1),4)for row in x_test.T])
normalized_x_test=(poly_x_test-min)/range
test_pred=myLR4.predict_(normalized_x_test)
mse=myLR4.mse_(y_test,test_pred)
print(f'testing mse on polynomial degree 4: {mse}.')
test_mse_list.append(mse)


myLR1.thetas.resize(13,1)
myLR2.thetas.resize(13,1)
myLR3.thetas.resize(13,1)
myLR4.thetas.resize(13,1)
models=pd.DataFrame(np.hstack([myLR1.thetas,myLR2.thetas,myLR3.thetas,myLR4.thetas])).to_csv('models.csv')