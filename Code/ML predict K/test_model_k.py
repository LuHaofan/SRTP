# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:48:44 2020

@author: lhf
"""

from sklearn.externals import joblib
import csv
import matplotlib.pyplot as plt
import numpy as np
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
Kp = [0.2, 0.5, 1, 2, 5, 10, 20]
Km = Kp
a = 0.3
b = 0.3
c = 0.49
f = (4/3)*np.pi*a*b*c
p = c/a

mlresult = np.zeros((len(Kp), len(Km)))
nanresult = np.zeros((len(Kp), len(Km)))
for i in range(len(Kp)):
    for j in range(len(Km)):
        nanresult[i,j] = Nan(Kp[i], Km[j], f, p)
        sample = np.array([[Kp[i], Km[j], f, p]])
        sample = scaler.transform(sample)
        pred = []
        for k in range(len(nn_model)):
            pred.append(nn_model[k].predict(sample))
        mlresult[i,j] = np.mean(pred)

diff = nanresult-mlresult
diff = np.reshape(diff, (49,))
plt.figure()
plt.plot(np.abs(diff))
plt.xlabel('sample')
plt.ylabel('Absolute error')
plt.title('Absolute Error in Training Range (maximum error:{:.2f})'.format(np.max(np.abs(diff))))
