#!/usr/bin/env python3
import sys
sys.path.append('../') # noqa: E402
import pandas as pd
import numpy as np
from ex08.my_logistic_regression import MyLogisticRegression as MyLR
from ex00.polynomial_model_extended import add_polynomial_features
from ex07.benchmark_train import data_spliter


def precision_score_(y,y_hat,pos_label=1):
    if not isinstance(y,np.ndarray)or not y.size:
        print("y has to be an numpy.array,a vector.")
        return None
    if not isinstance(y_hat,np.ndarray)or not y_hat.size:
        print("y_hat has to be an numpy.array,a vector.")
        return None
    if y.shape!=y_hat.shape:
        print('y and y_hat have different shapes')
        return None
    tp=((y_hat==pos_label)*(y==pos_label)).sum()
    fp=((y_hat==pos_label)*(1 - (y==pos_label))).sum()
    if tp+fp==0:
        return np.nan
    return tp/(tp+fp)

def recall_score_(y,y_hat,pos_label=1):
    if not isinstance(y,np.ndarray)or not y.size:
        print("y has to be an numpy.array,a vector.")
        return None
    if not isinstance(y_hat,np.ndarray)or not y_hat.size:
        print("y_hat has to be an numpy.array,a vector.")
        return None
    if y.shape!=y_hat.shape:
        print('y and y_hat have different shapes')
        return None
    tp=((y_hat==pos_label)*(y==pos_label)).sum()
    fn=((y_hat!=pos_label)*(y==pos_label)).sum()
    if tp+fn==0:
        return np.nan
    return tp/(tp+fn)

def f1_score_(y,y_hat,pos_label=1):
    if not isinstance(y,np.ndarray)or not y.size:
        print("y has to be an numpy.array,a vector.")
        return None
    if not isinstance(y_hat,np.ndarray)or not y_hat.size:
        print("y_hat has to be an numpy.array,a vector.")
        return None
    if y.shape!=y_hat.shape:
        print('y and y_hat have different shapes')
        return None
    precision=precision_score_(y,y_hat,pos_label)
    if precision is None:
        return None
    recall=recall_score_(y,y_hat,pos_label)
    if recall is None:
        return None
    if precision==np.nan or recall==np.nan:
        return np.nan
    return (2*precision*recall)/(precision+recall)

def one_vs_all(x_train,x_test,y_train,y_test,lambda_,categories,evaluate=True):
    theta=[]
    all_predictions=[]
    for zipcode,planet in enumerate(categories):
        # Train a logistic regression model to predict whether a citizen comes
        # from this zipcode or not
        new_y_train=(y_train==zipcode).astype(float)
        new_y_test=(y_test==zipcode).astype(float)
        mylr=MyLR(np.zeros((10,1)),max_iter=10**6,alpha=1e-1,lambda_=lambda_)
        mylr.fit_(x_train,new_y_train)
        theta.append(mylr.theta)
        train_predictions=mylr.predict_(x_train)
        print(f"Training set loss: {mylr.loss_(new_y_train,train_predictions)}")
        test_predictions=mylr.predict_(x_test)
        print(f"Test set loss: {mylr.loss_(new_y_test,test_predictions)}\n")
        all_predictions.append(test_predictions)
    if evaluate:
        # Evaluate the f1 score of the model for each category
        predictions=np.argmax(np.hstack(all_predictions),axis=1).reshape(-1,1)
        f1_scores=[]
        for zipcode,planet in enumerate(categories):
            f1_score=f1_score_(y_test,predictions,zipcode)
            print(f'f1 score on {planet}={f1_score}')
            f1_scores.append(f1_score)
        f1_scores=np.array(f1_scores)
        print(f'Average f1 score: {f1_scores.mean()}')
    return theta
if __name__=="__main__":
    # Load the dataset
    x=pd.read_csv("../resources/solar_system_census.csv",index_col=0)
    #  Labels
    y=pd.read_csv("../resources/solar_system_census_planets.csv",index_col=0)
    x=x.to_numpy()
    y=y.to_numpy()
    categories=["The flying cities of Venus","United Nations of Earth","Mars Republic","The Asteroids' Belt colonies"]
    # We will use a polynomial hypothesis of degree 3
    x_poly=add_polynomial_features(x,3)
    # Normalization
    min=x_poly.min(axis=0)
    range=x_poly.max(axis=0)- min
    x_poly=(x_poly - min)/range
    # Split the dataset into a training,a validation,and a test set
    (x_rest,x_test,y_rest,y_test)=data_spliter(x_poly,y,0.8)
    (x_train,x_valid,y_train,y_valid)=data_spliter(x_rest,y_rest,0.8)
    # Train different regularized logistic regression
    # models with a polynomial hypothesis of degree 3. The models will be
    # trained with different λ values,ranging from 0 to 1
    lambdas=[0,0.01,0.05,0.1,0.3]
    models=[]
    for lambda_ in lambdas:
        theta=one_vs_all(x_train,x_valid,y_train,y_valid,lambda_,categories)
        models.extend(theta)
    # Evaluate the best model on the test set
    print("Evaluating the best model on the test set:")
    best_theta=models[4*2:4*2+4]
    all_predictions=[]
    for theta in best_theta:
        my_lr=MyLR(theta)
        predictions=my_lr.predict_(x_test)
        all_predictions.append(predictions)
    predictions=np.argmax(np.hstack(all_predictions),axis=1).reshape(-1,1)
    f1_scores=[]
    for zipcode,planet in enumerate(categories):
        f1_score=f1_score_(y_test,predictions,zipcode)
        print(f'f1 score on {planet}={f1_score}')
        f1_scores.append(f1_score)
    f1_scores=np.array(f1_scores)
    print(f'Average f1 score: {f1_scores.mean()}')
    # Save the parameters of the different models into a file
    all_models=np.hstack(models)
    pd.DataFrame(all_models).to_csv("models.csv",float_format='%07.3f',index=False)