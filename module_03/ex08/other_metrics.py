import numpy as np

def accuracy_score_(y,y_hat):
    if not isinstance(y,np.ndarray)or not y.size:
        print('y has to be an numpy array.')
        return None
    if not isinstance(y_hat,np.ndarray)or not y_hat.size:
        print('y_hat has to be an numpy array.')
        return None
    if y.shape != y_hat.shape:
        print('y and y_hat has different shapes.')
        return None
    return (y==y_hat).mean()

def precision_score_(y,y_hat,pos_label=1):
    if not isinstance(y,np.ndarray)or not y.size:
        print('y has to be an numpy array.')
        return None
    if not isinstance(y_hat,np.ndarray)or not y_hat.size:
        print('y_hat has to be an numpy array.')
        return None
    if y.shape != y_hat.shape:
        print('y and y_hat has different shapes.')
        return None
    tp=((y_hat==pos_label)*(y==pos_label)).sum()
    fp=((y_hat==pos_label)*(1-(y==pos_label))).sum()
    return tp/(tp+fp)

def recall_score_(y,y_hat,pos_label=1):
    if not isinstance(y,np.ndarray)or not y.size:
        print('y has to be an numpy array.')
        return None
    if not isinstance(y_hat,np.ndarray)or not y_hat.size:
        print('y_hat has to be an numpy array.')
        return None
    if y.shape != y_hat.shape:
        print('y and y_hat has different shapes.')
        return None
    tp=((y_hat==pos_label)*(y==pos_label)).sum()
    fn=((y_hat!=pos_label)*(y==pos_label)).sum()
    return tp/(tp+fn)

def f1_score_(y,y_hat,pos_label=1):
    if not isinstance(y,np.ndarray)or not y.size:
        print('y has to be an numpy array.')
        return None
    if not isinstance(y_hat,np.ndarray)or not y_hat.size:
        print('y_hat has to be an numpy array.')
        return None
    if y.shape != y_hat.shape:
        print('y and y_hat has different shapes.')
        return None
    prec=precision_score_(y,y_hat,pos_label)
    if prec is None:
        return None
    recall=recall_score_(y,y_hat,pos_label)
    if recall is None:
        return None
    return (2*prec*recall)/(prec+recall)