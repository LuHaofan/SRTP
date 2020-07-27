# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:24:53 2020

@author: lhf
"""
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import xlrd
#load model and scaler
modelname = '../../Model/FindKnnmodel4f_v1'
nn_model = joblib.load(modelname)
scalername = '../../Model/FindK4fscaler'
scaler = joblib.load(scalername)
def Nan(Kp, Km, f, p):
    if p > 1:
        L1 = p*p/(2*(p*p-1))-p*np.arccosh(p)/(2*(p*p-1)**(1.5))
    elif p < 1:
        L1 = p*p/(2*(p*p-1))+p*np.arccos(p)/(2*(1-p*p)**(1.5))
    else:
        L1 = 1/3
    L2 = L1
    L3 = 1-2*L1
    L = np.array([L1, L2, L3])
    Kc = Kp
    beta = (Kc-Km)/(Km+L*(Kc-Km))
    return Km*(3+f*(2*beta[0]*(1-L[0])+beta[2]*(1-L[2])))/(3-f*(2*beta[0]*L[0]+beta[2]*L[2]));

data = xlrd.open_workbook('source.xlsx')
table = data.sheet_by_name('Sheet1')
fe = table.col_values(0)[1:]
f = 0.046176681
p = 0.3129/0.1877
Kp = [0.2, 0.5, 1, 2, 5, 10, 20]
Km = Kp
nan = []
for i in range(len(Km)):
    for j in range(len(Kp)):
        nan.append(Nan(Kp[j], Km[i], f, p))
        
ml = []
for i in range(len(Km)):
    for j in range(len(Kp)):
        sample = np.array([[Kp[j], Km[i], f, p]])
        sample = scaler.transform(sample)
        pred = []
        for k in range(len(nn_model)):
            pred.append(nn_model[k].predict(sample))
        ml.append(np.mean(pred))
            
plt.figure(figsize = (15,6))
x = np.linspace(0, 20, 100)
y = x
plt.subplot(121)
plt.plot(x,y, 'r-', label = 'y = x')
plt.scatter(fe, nan)
plt.xlabel('Finite Element')
plt.ylabel('EMA Formula')
plt.title('Compare EMA formula with Finite Element')
plt.legend()

plt.subplot(122)
plt.plot(x, y, 'r-', label = 'y = x')
plt.scatter(fe, ml, marker = '^')
plt.ylabel('Machine Learning')
plt.xlabel('Finite Element')
plt.title('Compare Machine Learning with Finite Element')
plt.legend()


        