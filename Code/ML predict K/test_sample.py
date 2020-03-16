# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:15:14 2019

@author: Lu Haofan
"""

from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import csv
#load model and scaler
modelname = 'nnmodel4f_v1.2'
nn_model = joblib.load(modelname)
scalername = '4fscaler'
scaler = joblib.load(scalername)
#%% Test a simple sample
sample_realp = np.array([[10,5,0.046181411,0.4023]])
sample_realp = scaler.transform(sample_realp)
avg = []
for m in range(10):
    pred_realp = nn_model[m].predict(sample_realp)
    avg.append(pred_realp)
print(sum(avg)/10)
#%% Test different p

sample = np.array([[2,1,0.1847,p] for p in np.linspace(0,10,100)])
sample = scaler.transform(sample)
pred = nn_model.predict(sample)
p = np.linspace(0,10,100)
for i in range(100):
    print("{}:{}".format(p[i], np.abs(pred[i]-1.1458)))
    
plt.figure()
plt.plot(p, np.abs(pred-1.1458))
plt.xlabel('P')
plt.ylabel('K')
plt.show()

#%%Test for Sphere
f = 0.268082569
Sphericity = 1
p = 1
Kp = [0.1,0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
Km = Kp
filename = '4fTestResults.csv'
with open(filename, 'a', newline='') as file: 
    f_csv = csv.writer(file)
    rows = []
    for j in range(len(Km)):
        for i in range(len(Kp)):
            sample = np.array([[Kp[i], Km[j], f, p]])
            sample = scaler.transform(sample)
            pred = nn_model.predict(sample)
            rows.append((Kp[i], Km[j],f, p, pred[0]))
    f_csv.writerows(rows)

#%% Test Cone
h = 0.49
r = 0.3
l = np.sqrt(h**2+r**2)
S = np.pi*r**2+np.pi*r*l
V = (1/3)*r**2*np.pi*h
AvgR = np.sqrt(3*V/np.pi)
f = 0.067020642
Sphericity=((2*p)**(2/3))/(1+np.sqrt(1+p**2))
Kp = [0.1,0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
Km = Kp
filename = '5fTestResults.csv'
with open(filename, 'a', newline='') as file: 
    f_csv = csv.writer(file)
    rows = []
    for j in range(len(Km)):
        for i in range(len(Kp)):
            sample = np.array([[Kp[i], Km[j], f, Sphericity, p]])
            sample = scaler.transform(sample)
            pred = nn_model.predict(sample)
            rows.append((Kp[i], Km[j],f, Sphericity, p, pred[0]))
    f_csv.writerows(rows)


#%%Compute for donut
R = 0.3
r = 0.19
f = 0.213775624

def computeSphericity(R, r):
    p = (r/R)**(-1)
    v = 2*(np.pi**2)*R*(r**2)
    s = 4*(np.pi**2)*R*r
    sphericity = (np.pi**(1/3))*(6*v)**(2/3)/s
    return p, sphericity
Kp = [0.1,0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
Km = Kp
filename = '4fTestResults.csv'
with open(filename, 'a', newline='') as file: 
    f_csv = csv.writer(file)
    rows = []
    for j in range(len(Km)):
        for i in range(len(Kp)):
            p, Sphericity = computeSphericity(R,r)
            sample = np.array([[Kp[i], Km[j], f,p]])
            sample = scaler.transform(sample)
            pred = nn_model.predict(sample)
            rows.append((Kp[i], Km[j],f,p, pred[0]))
    f_csv.writerows(rows)

#%% Kp, Km, f, Sphericity,p
for i in range(2):
    sample = np.array([[2, 80, 0.0083775803, Sphericity[i], p[i]],[20, 80, 0.0083775803, Sphericity[i], p[i]]])
    sample = scaler.transform(sample)
    pred = nn_model.predict(sample)
    print("Sphericity:{}  p:{}".format(Sphericity[i],p[i]))
    print("{}:{}".format(2,pred[0]))
    print("{}:{}".format(20,pred[1]))
    print("\n")

#%% compute the k for Donuts
r = 0.19
def computeSphericity(R, r):
    p = (r/R)**(-1)
    v = 2*(np.pi**2)*R*(r**2)
    s = 4*(np.pi**2)*R*r
    sphericity = (np.pi**(1/3))*(6*v)**(2/3)/s
    return p, sphericity
R = 0.3
f = 0.213776


kp = 20
km = [0.1,1,100,200,300,400]

km = [0.1,0.1,0.1,1,1,1,5,5,5]
kp = [0.1,1,5,0.1,1,5,0.1,1,5]
for i in range(len(km)):
    p, sph = computeSphericity(R,r)
    sample = np.array([[kp[i], km[i], f,sph, p]])
    sample = scaler.transform(sample)
    pred = nn_model.predict(sample)
    print("Kp:{}\tKm:{}\tf:{}\tsphericity:{}\tp:{}\tresult:{}".format(kp[i], km[i],f, sph, p, pred[0]))
   

