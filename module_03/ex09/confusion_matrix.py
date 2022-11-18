import numpy as np
import pandas as pd

def confusion_matrix_(y_true,y_hat,labels=None,df_option=False):
    if not isinstance(y_true,np.ndarray)or not y_true.size:
        print('y has to be an numpy array.')
        return None
    if not isinstance(y_hat,np.ndarray)or not y_hat.size:
        print('y_hat has to be an numpy array.')
        return None
    if y_true.shape != y_hat.shape:
        print('y and y_hat has different shapes.')
        return None
    if labels is not None and not isinstance(labels,list):
        print('labels has to be None or a list.')
        return None
    if labels is None:
        labels=sorted(list(set(np.concatenate((y_true,y_hat)).ravel())))
    else:
        labels=sorted(labels)
    cols=[]
    for label in labels:
        value_counts=dict(zip(labels,[0]*len(labels)))
        idx=y_hat==label
        correct_labels=y_true[idx]
        unique,counts=np.unique(correct_labels,return_counts=True)
        correct_labels_counts=dict(zip(unique,counts))
        value_counts.update((label,correct_labels_counts[label])for label in value_counts.keys() & correct_labels_counts.keys())
        col=np.array(list(value_counts.values())).reshape(-1,1)
        cols.append(col)
    confusion_matrix=np.hstack(cols)
    if df_option:
        return pd.DataFrame(confusion_matrix,columns=labels,index=labels)
    else:
        return confusion_matrix