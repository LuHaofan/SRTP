# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:29:22 2020

@author: lhf
"""

import numpy as np
import csv
from sklearn.externals import joblib
import matplotlib.pyplot as plt

#load model and scaler
modelname = '../../Model/nnmodelFP_v7'
nn_model = joblib.load(modelname)
scalername = '../../Model/FindPscaler_v7'
scaler = joblib.load(scalername)

#%% load test data
fname = '../../Data/P_testset_pro.csv'
print('Loading data ...\n')
X = []
y = []
with open(fname) as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for row in csv_reader:
        X.append(row[3:9])
        y.append(row[9])
        
X = [[float(i) for i in row] for row in X]
y = [float(j) for j in y]

#%% test the model
Test_X = np.array(X[:])
Test_y = np.array(y[:])
Test_X = scaler.transform(Test_X)

temp = np.array([0]*Test_y.shape[0])
for m in range(10):
    pred_y = nn_model[m].predict(Test_X)
    temp = temp + pred_y
#%% Compute cost
temp = temp/10
error = np.abs(Test_y-temp)
cost = np.sum(error**2)/len(error)

#%% Relative Error
relative_error = error/Test_y
#%% Plot
cnt = 0

plt.figure(figsize = (7, 6))
plt.hist(error, bins = 40, range = (0, 0.05))
plt.xlabel('Absolute Error')
plt.ylabel('Number of samples')
#plt.text(0.05, 0.5, 'P{e < 0.1} = 0.99')
plt.title('Distribution of Absolute Error')
for e in error:
    if e < 0.01:
        cnt +=1
print(cnt/len(Test_y))
