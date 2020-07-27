# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 15:31:30 2020

@author: lhf
"""

import numpy as np
import csv
from sklearn.externals import joblib
import matplotlib.pyplot as plt
#%%
modelname = '../../Model/FindKnnmodel4f_v1'
nn_model = joblib.load(modelname)
scalername = '../../Model/FindK4fscaler'
scaler = joblib.load(scalername)

fname = r'C:\Users\lhf\Documents\SRTP_Repo\SRTP\Data\K_trainset_pro.csv'
print('Loading data ...\n')
X = []
y = []
with open(fname) as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for row in csv_reader:
        X.append(row[0:4])
        y.append(row[4])
        
X = [[float(i) for i in row] for row in X]
y = [float(j) for j in y]

Training_X = np.array(X[:])
Training_y = np.array(y[:])
Training_X = scaler.transform(Training_X) 
#%% Predict
temp = np.array([0]*Training_y.shape[0])
for k in range(10):
    pred_train = nn_model[k].predict(Training_X)
    temp = temp + pred_train

#%% Compute cost
temp = temp/10
error = np.abs(Training_y-temp)
cost = np.sum(error**2)/len(error)

#%% Relative Error
relative_error = error/Training_y

#%% Plot
cnt = 0

plt.figure(figsize = (15, 6))
plt.subplot(121)
plt.hist(error, bins = 40, range = (0, 0.2))
plt.xlabel('Absolute Error')
plt.ylabel('Number of samples')
#plt.text(0.05, 0.5, 'P{e < 0.1} = 0.99')
plt.title('Distribution of Absolute Error')
for e in error:
    if e < 0.1:
        cnt +=1
print(cnt/10000)
plt.subplot(122)
plt.hist(relative_error, bins = 50, range = (0,0.05))
plt.xlabel('Relative Error')
plt.ylabel('Number of samples')
#plt.text(0.05, 0.5, 'P{e < 0.01} = 0.93')
plt.title('Distribution of Relative Error')
cnt = 0
for e in relative_error:
    if e < 0.01:
        cnt +=1
print(cnt/10000)