# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:14:21 2020

@author: lhf
"""
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.externals import joblib
import csv
import random
fname = '../../../code/EllipsoidTrain/P_trainset.csv'
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
#%%  Normalize data
print('Normalizing ...\n')
Training_X = np.array(X[:])
Training_y = np.array(y[:])
scaler = StandardScaler()  
scaler.fit(Training_X)  
Training_X = scaler.transform(Training_X)  

#%% train model
print('Training model...')
model = []
for i in range(10):
    print('Training Model {}'.format(i))
    est = MLPRegressor(hidden_layer_sizes=(10,25,25,25,10), alpha = 0, activation='relu', verbose = True,
                           learning_rate_init = 0.001, random_state = random.seed(0), max_iter = 1000)
    est.fit(Training_X, Training_y)
    model.append(est)
'''
est = MLPRegressor(hidden_layer_sizes=(10,25,25,25,10), alpha = 0, activation='relu', verbose = True,
                   learning_rate_init = 0.001, random_state = random.seed(0), max_iter = 1000)
est.fit(Training_X, Training_y)
'''
#%% Save the model and scaler
joblib.dump(scaler, 'FindPscaler_v2')
joblib.dump(model, filename='nnmodelFP_v2')