from cgi import test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MLR
from data_spliter import data_spliter

#load csv
df=pd.read_csv('../resources/space_avocado.csv',index_col=0)
x=np.array(df[['weight','prod_distance','time_delivery']])
y=np.array(df['target']).reshape(-1,1)
(x_train,x_test,y_train,y_test)=data_spliter(x,y,.8)

#fetching saved models
models=pd.read_csv('./models.csv',index_col=0).to_numpy()
theta1=models[:4,0].reshape(-1,1)
theta2=models[:7,1].reshape(-1,1)
theta3=models[:10,2].reshape(-1,1)
theta4=models[:,3].reshape(-1,1)

training_mse_lst=[]
testing_mse_lst=[]

#run predictions of polynomial degree 1
#on training set
myLR1=MLR(theta1)
min=x_train.min(axis=0)
range=x_train.max(axis=0)-min
normalized_x_train=(x_train-min)/range
pred=myLR1.predict_(normalized_x_train)
mse=myLR1.mse_(y_train,pred)
training_mse_lst.append(mse)
#on testing set
normalized_x_test=(x_test-min)/range
pred=myLR1.predict_(normalized_x_test)
mse=myLR1.mse_(y_test,pred)
testing_mse_lst.append(mse)

#run predictions of polynomial degree 2
#on training set
poly_x_train = np.hstack([add_polynomial_features(row.reshape(-1, 1), 2)for row in x_train.T])
min=poly_x_train.min(axis=0)
range=poly_x_train.max(axis=0)-min
normalized_poly_x_train=(poly_x_train-min)/range
myLR2=MLR(theta2)
myLR2.fit_(normalized_poly_x_train,y_train)
pred=myLR2.predict_(normalized_poly_x_train)
mse=myLR2.mse_(y_train,pred)
training_mse_lst.append(mse)
#on testing set
poly_x_test = np.hstack([add_polynomial_features(row.reshape(-1, 1), 2)for row in x_test.T])
normalized_x_test=(poly_x_test-min)/range
test_pred=myLR2.predict_(normalized_x_test)
mse=myLR2.mse_(y_test,test_pred)
testing_mse_lst.append(mse)

#run predictions of polynomial degree 3
#on training set
poly_x_train = np.hstack([add_polynomial_features(row.reshape(-1,1),3)for row in x_train.T])
min=poly_x_train.min(axis=0)
range=poly_x_train.max(axis=0)-min
normalized_poly_x_train=(poly_x_train-min)/range
myLR3=MLR(theta3)
myLR3.fit_(normalized_poly_x_train,y_train)
training_pred=myLR3.predict_(normalized_poly_x_train)
mse=myLR3.mse_(y_train,training_pred)
training_mse_lst.append(mse)
#on testing set
poly_x_test = np.hstack([add_polynomial_features(row.reshape(-1,1),3)for row in x_test.T])
normalized_x_test=(poly_x_test-min)/range
testing_pred=myLR3.predict_(normalized_x_test)
mse=myLR3.mse_(y_test,testing_pred)
testing_mse_lst.append(mse)

#run predictions of polynomial degree 4
#on training set
poly_x_train = np.hstack([add_polynomial_features(row.reshape(-1,1),4)for row in x_train.T])
min=poly_x_train.min(axis=0)
range=poly_x_train.max(axis=0)-min
normalized_poly_x_train=(poly_x_train-min)/range
myLR4=MLR(theta4)
myLR4.fit_(normalized_poly_x_train,y_train)
pred=myLR4.predict_(normalized_poly_x_train)
mse=myLR4.mse_(y_train,pred)
training_mse_lst.append(mse)
#on testing set
poly_x_test = np.hstack([add_polynomial_features(row.reshape(-1,1),4)for row in x_test.T])
normalized_x_test=(poly_x_test-min)/range
test_pred=myLR4.predict_(normalized_x_test)
mse=myLR4.mse_(y_test,test_pred)
testing_mse_lst.append(mse)

#plot mean squared error depending of the polynomial degree
x_axis_ticks=np.arange(1,5)
plt.bar(x_axis_ticks-.1,training_mse_lst,label='Training set MSE',width=.2)
plt.bar(x_axis_ticks+.1,testing_mse_lst,label='Testing set MSE',width=.2)
plt.xticks(x_axis_ticks)
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

#plot real values and predicted values
#weight values
data=np.vstack((x_train,x_test))
predictions=np.vstack((training_pred,testing_pred))
x=np.array(df['weight']).reshape(-1,1)
plt.scatter(x,y,label='True values')
x_=data[:,0].reshape(-1,1)
plt.scatter(x_,predictions,label='Predictions')
plt.xlabel('weight(in tons)')
plt.ylabel('target(in trantorian unit')
plt.legend()
plt.show()
#prod_distance values
x=np.array(df['prod_distance']).reshape(-1,1)
plt.scatter(x,y,label='True values')
x_=data[:,1].reshape(-1,1)
plt.scatter(x_,predictions,label='Predictions')
plt.xlabel('prod_distance(in Mkm)')
plt.ylabel('target(in trantorian unit')
plt.legend()
plt.show()
#time_delivery values
x=np.array(df['time_delivery']).reshape(-1,1)
plt.scatter(x,y,label='True values')
x_=data[:,2].reshape(-1,1)
plt.scatter(x_,predictions,label='Predictions')
plt.xlabel('time_delivery(in days)')
plt.ylabel('target(in trantorian unit')
plt.legend()
plt.show()