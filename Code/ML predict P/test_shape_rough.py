# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:53:18 2020

@author: lhf
"""

from sklearn.externals import joblib
import numpy as np
import csv

#%%load model and scaler
modelname = '../../Model/nnmodel4f_predK'
nn_model = joblib.load(modelname)
scalername = '../../Model/4fscaler'
scaler = joblib.load(scalername)



#%% Test Cone

h = 0.49
r = 0.3

l = np.sqrt(h**2+r**2)
S = np.pi*r**2+np.pi*r*l
V = (1/3)*r**2*np.pi*h
f = V

p = 2*h/r
Kp = [0.2, 0.5, 1, 2, 5, 10, 20]
Km = Kp
filename = 'cone_param_old4.csv'
with open(filename, 'w', newline='') as file: 
    f_csv = csv.writer(file)
    rows = []
    for j in range(len(Km)):
        for i in range(len(Kp)):
            sample = np.array([[Kp[i], Km[j], f, p]])
            sample = scaler.transform(sample)
            pred = nn_model.predict(sample)
            rows.append((Kp[i], Km[j],f, p, pred[0]))
    f_csv.writerows(rows)

#%% Test Ellipsoid
a = 0.1877
b = 0.1877
c = 0.3129
V = (4/3)*np.pi*a*b*c
f = V
p = c/a

Kp = [0.2, 0.5, 1, 2, 5, 10, 20]
Km = Kp
filename = 'ellipsoid_param_old4.csv'
with open(filename, 'w', newline='') as file: 
    f_csv = csv.writer(file)
    rows = []
    for j in range(len(Km)):
        for i in range(len(Kp)):
            sample = np.array([[Kp[i], Km[j], f, p]])
            sample = scaler.transform(sample)
            pred = nn_model.predict(sample)
            rows.append((Kp[i], Km[j],f, p, pred[0]))
    f_csv.writerows(rows)
#%% Test Cube
    
#%% Test Donut
R = 0.1866
r = 0.112
V = 2*np.pi**2*R*r**2
S = 4*np.pi**2*R*r
f = V

p = R/r
Kp = [0.2, 0.5, 1, 2, 5, 10, 20]
Km = Kp
filename = 'donut_param_old4.csv'
with open(filename, 'w', newline='') as file: 
    f_csv = csv.writer(file)
    rows = []
    for j in range(len(Km)):
        for i in range(len(Kp)):
            sample = np.array([[Kp[i], Km[j], f, p]])
            sample = scaler.transform(sample)
            pred = nn_model.predict(sample)
            rows.append((Kp[i], Km[j],f, p, pred[0]))
    f_csv.writerows(rows)