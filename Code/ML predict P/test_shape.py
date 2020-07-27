# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:53:18 2020

@author: lhf
"""

from sklearn.externals import joblib
import numpy as np
import csv

#%%load model and scaler
modelname = '../../Model/FindKnnmodel4f_v1'
nn_model = joblib.load(modelname)
scalername = '../../Model/FindK4fscaler'
scaler = joblib.load(scalername)
pmodelname = '../../Model/nnmodelFP_v7'
p_model = joblib.load(pmodelname)
pscalername = '../../Model/FindPscaler_v7'
p_scaler = joblib.load(pscalername)


#%% Test Cone

h = 0.49
r = 0.3

l = np.sqrt(h**2+r**2)
S = np.pi*r**2+np.pi*r*l
V = (1/3)*r**2*np.pi*h
f = V
AvgR = (3*V/4/np.pi)**(1/3)
Pro_1 = h*r
Pro_2 = h*r
Pro_3 = r**2*np.pi

p_var = p_scaler.transform([[V, S, Pro_1, Pro_2, Pro_3, AvgR]])
pavg = []
for i in range(10):
    predp = p_model[i].predict(p_var)
    pavg.append(predp[0])
p = sum(pavg)/10

#p = 2*h/r
Kp = [0.2, 0.5, 1, 2, 5, 10, 20]
Km = Kp
filename = 'cone_param_pro5.csv'
with open(filename, 'w', newline='') as file: 
    f_csv = csv.writer(file)
    rows = []
    for j in range(len(Km)):
        for i in range(len(Kp)):
            sample = np.array([[Kp[i], Km[j], f, p]])
            sample = scaler.transform(sample)
            pred = []
            for k in range(len(nn_model)):
                pred.append(nn_model[k].predict(sample))
            rows.append((Kp[i], Km[j],f, p, np.mean(pred)))
    f_csv.writerows(rows)

#%% Test Ellipsoid
a = 0.45
b = 0.45
c = 0.49
V = (4/3)*np.pi*a*b*c
f = V
AvgR = (3*V/4/np.pi)**(1/3)
Pro_1 = np.pi*b*c
Pro_2 = np.pi*b*a
Pro_3 = np.pi*a*c
S = 4*np.pi*((a**1.6*b**1.6+a**1.6*c**1.6+b**1.6*c**1.6)/3)**(1/1.6)
p_var = p_scaler.transform([[V, S, Pro_1, Pro_2, Pro_3, AvgR]])
pavg = []
for i in range(10):
    predp = p_model[i].predict(p_var)
    pavg.append(predp[0])
p = sum(pavg)/10
#%%
Kp = [0.2, 0.5, 1, 2, 5, 10, 20]
Km = Kp
filename = 'ellipsoid_param_pro6.csv'
with open(filename, 'w', newline='') as file: 
    f_csv = csv.writer(file)
    rows = []
    for j in range(len(Km)):
        for i in range(len(Kp)):
            sample = np.array([[Kp[i], Km[j], f, p]])
            sample = scaler.transform(sample)
            pred = []
            for k in range(len(nn_model)):
                pred.append(nn_model[k].predict(sample))
            rows.append((Kp[i], Km[j],f, p, np.mean(pred)))
    f_csv.writerows(rows)
#%% Test helix
n_turns = 3
R = 0.1574
r = 0.07
pitch = 0.15
V = np.sqrt((2*R*np.pi)**2+pitch**2)*n_turns*(r**2*np.pi)
S = np.sqrt((2*R*np.pi)**2+pitch**2)*n_turns*(2*r*np.pi) + 2*r**2*np.pi
f = V
Pro_1 = (r**2*np.pi+4*R*r)*n_turns+0.01*4*R
Pro_2 = Pro_1
Pro_3 = (R+r)**2*np.pi - (R-r)**2*np.pi
AvgR = (3*V/4/np.pi)**(1/3)
p_var = p_scaler.transform([[V, S, Pro_1, Pro_2, Pro_3, AvgR]])
pavg = []
for i in range(10):
    predp = p_model[i].predict(p_var)
    pavg.append(predp[0])
p = sum(pavg)/10

Kp = [0.2, 0.5, 1, 2, 5, 10, 20]
Km = Kp
filename = 'helix_param_pro7.csv'
with open(filename, 'w', newline='') as file: 
    f_csv = csv.writer(file)
    rows = []
    for j in range(len(Km)):
        for i in range(len(Kp)):
            sample = np.array([[Kp[i], Km[j], f, p]])
            sample = scaler.transform(sample)
            pred = []
            for k in range(len(nn_model)):
                pred.append(nn_model[k].predict(sample))
            rows.append((Kp[i], Km[j],f, p, np.mean(pred)))
    f_csv.writerows(rows)
    
#%% Test Donut
R = 0.25
r = 0.24
V = 2*np.pi**2*R*r**2
S = 4*np.pi**2*R*r
f = V
Pro_1 = 2*(R+r)*2*r
Pro_2 = 2*(R+r)*2*r
Pro_3 = (R+r)**2*np.pi - (R-r)**2*np.pi
AvgR = (3*V/4/np.pi)**(1/3)
p_var = p_scaler.transform([[V, S, Pro_1, Pro_2, Pro_3, AvgR]])
pavg = []
for i in range(10):
    predp = p_model[i].predict(p_var)
    pavg.append(predp[0])
p = sum(pavg)/10
#%%
p = 9
#p = R/r
Kp = [0.2, 0.5, 1, 2, 5, 10, 20]
Km = Kp
filename = 'donut_param_pro6.csv'
with open(filename, 'w', newline='') as file: 
    f_csv = csv.writer(file)
    rows = []
    for j in range(len(Km)):
        for i in range(len(Kp)):
            sample = np.array([[Kp[i], Km[j], f, p]])
            sample = scaler.transform(sample)
            pred = []
            for k in range(len(nn_model)):
                pred.append(nn_model[k].predict(sample))
            rows.append((Kp[i], Km[j],f, p, np.mean(pred)))
    f_csv.writerows(rows)
    
