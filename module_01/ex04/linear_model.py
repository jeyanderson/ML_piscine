import numpy as np
from my_linear_regression import MyLinearRegression as MLR
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('../resources/are_blue_pills_magics.csv',index_col=0)
df=df.to_numpy()
dose=df[:,0].reshape(-1,1)
perf=df[:,1].reshape(-1,1)
lr=MLR([0,0],max_iter=50000)
lr.fit_(dose,perf)
thetas=lr.thetas
print(f'thetas: {repr(thetas)}.')
pred=lr.predict_(dose)
print(f'predictions: {repr(pred)}.')
print(f'MSE: {lr.mse_(dose,pred)}.')
plt.grid()
plt.scatter(dose,perf,color='deepskyblue')
plt.scatter(dose,pred,color='chartreuse')
plt.plot(dose,pred,color='chartreuse',linestyle='dashed')
plt.show()
plt.grid()
theta_0_s=np.linspace(77, 97, 6)
theta_1_s=np.linspace(-15, -3, 100)
for theta_0 in theta_0_s:
    J=[]
    for theta_1 in theta_1_s:
        lr = MLR([theta_0, theta_1],max_iter=50000)
        predictions=lr.predict_(dose)
        J.append(lr.loss_(perf, predictions))
    plt.plot(theta_1_s, J,label=f'J(θ_0={theta_0}, θ_1)')
plt.xlim((-15, -3))
plt.ylim((10, 150))
plt.xlabel('θ_1')
plt.ylabel('cost function J(θ_0, θ_1)')
plt.legend(loc='lower right')
plt.show()