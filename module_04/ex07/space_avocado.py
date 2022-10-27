import sys
sys.path.append('../')  # noqa: E402
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ex06.ridge import MyRidge as MyR
from benchmark_train import data_spliter
from ex00.polynomial_model_extended import add_polynomial_features


# Load the dataset
df=pd.read_csv("../resources/space_avocado.csv",index_col=0)
x=np.array(df[['weight','prod_distance','time_delivery']])
n_features=x.shape[1]
y=np.array(df['target']).reshape(-1,1)
(x_train,x_test,y_train,y_test)=data_spliter(x,y,0.8)

degree=4
my_lr=MyR(np.ones((1+n_features*degree,1)),1e-1,10**4,0)
x_poly=add_polynomial_features(x_train, degree)


# Normalization
min=x_poly.min(axis=0)
rnge=x_poly.max(axis=0)-min
x_poly=(x_poly-min)/rnge

my_lr.fit_(x_poly,y_train)
train_predictions=my_lr.predict_(x_poly)
train_mse=my_lr.mse_(y_train,train_predictions)
print(f'Train set mse: {train_mse}.')


# Evaluate model on the test set
x_poly=add_polynomial_features(x_test,degree)
x_poly=(x_poly-min)/rnge
test_predictions=my_lr.predict_(x_poly)
test_mse=my_lr.mse_(y_test,test_predictions)
print(f'Test set mse: {test_mse}.')


# Load models saved from benchmark_train.py
models=pd.read_csv("models.csv").to_numpy()


# Evaluate all the models we've trained in benchmark_train.py
degrees=list(range(1,5))
lambdas=np.arange(0,1,0.2).tolist()

evaluation_data=[]
for degree in degrees:
    for i,lambda_ in enumerate(lambdas):
        n_rows=1+n_features * degree
        col=(degree-1)*len(lambdas) + i
        saved_theta=models[:n_rows,col].copy().reshape(-1,1)
        my_lr=MyR(saved_theta)
        # Evaluate model on the test set
        x_poly=add_polynomial_features(x_test,degree)

        # Normalization
        min=x_poly.min(axis=0)
        range=x_poly.max(axis=0) - min
        x_poly=(x_poly - min) / range
        predictions=my_lr.predict_(x_poly)
        test_mse=my_lr.mse_(y_test, predictions)
        print(f'Test set mse: {test_mse}.')
        evaluation_data.append([degree,lambda_,test_mse])


# Plot a bar plot showing the MSE score of the models depending of the
# polynomial degree of the hypothesis and the regularization factor
data_df=pd.DataFrame(evaluation_data,columns=['Degree', 'Lambda','Test Set Mean Squared Error'])
sns.set_style("darkgrid")
sns.catplot(data=data_df,x='Lambda',y='Test Set Mean Squared Error',col='Degree',kind='bar')
plt.show()