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
pmodelname = '../../Model/nnmodelFP_predPavg'
p_model = joblib.load(pmodelname)
pscalername = '../../Model/FindPscaler'
p_scaler = joblib.load(pscalername)


#%% Test Cone
f = 0.046181411
h = 0.49
r = 0.3
'''
l = np.sqrt(h**2+r**2)
S = np.pi*r**2+np.pi*r*l
V = (1/3)*r**2*np.pi*h
AvgR = (3*V/np.pi)**(1/3)
p_var = p_scaler.transform([[V, S, AvgR]])
pavg = []
for i in range(10):
    predp = p_model[i].predict(p_var)
    pavg.append(predp[0])
p = sum(pavg)/10
'''
p = h/r
Kp = [0.2, 0.5, 1, 2, 5, 10, 20]
Km = Kp
filename = 'cone_rough_p.csv'
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
a = 0.49
b = 0.3
c = 0.3
V = (4/3)*np.pi*a*b*c
f = V
'''
e2 = 1-c**2/a**2
S = 2*np.pi*a**2*(1+(c**2*np.arctanh(e2**0.5))/(e2**0.5*a**2))
AvgR = (3*V/np.pi)**(1/3)
p_var = p_scaler.transform([[V, S, AvgR]])
pavg = []
for i in range(10):
    predp = p_model[i].predict(p_var)
    pavg.append(predp[0])
p = sum(pavg)/10
'''
p = c/a
Kp = [0.2, 0.5, 1, 2, 5, 10, 20]
Km = Kp
filename = 'ellipsoid_rough_p.csv'
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
R = 0.3
r = 0.19
V = 2*np.pi**2*R*r**2
S = 4*np.pi**2*R*r
f = 0.213775624
'''
AvgR = (3*V/np.pi)**(1/3)
p_var = p_scaler.transform([[V, S, AvgR]])
pavg = []
for i in range(10):
    predp = p_model[i].predict(p_var)
    pavg.append(predp[0])
p = sum(pavg)/10
'''
p = R/r
Kp = [0.2, 0.5, 1, 2, 5, 10, 20]
Km = Kp
filename = 'donut_rough_p.csv'
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