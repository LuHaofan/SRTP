# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:31:11 2019

@author: lhf
"""

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import numpy as np
import csv
import random
from sklearn.externals import joblib

fname = 'randEllipsoid_4f_0~100.csv'
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

# divide the Training, Cross Validation and Test sets
print('Normalizing ...\n')
Training_X = np.array(X[:])
Training_y = np.array(y[:])
scaler = StandardScaler()  
scaler.fit(Training_X)  
Training_X = scaler.transform(Training_X)  
# only train model
print('Training model...')
model = []
for i in range(10):
    est = MLPRegressor(hidden_layer_sizes=(10,25,25,25,10), alpha = 0, activation='relu', verbose = True,
                           learning_rate_init = 0.001, random_state = random.seed(0), max_iter = 1000)
    est.fit(Training_X, Training_y)
    model.append(est)
# save the model and scaler
joblib.dump(scaler, '4fscaler')
joblib.dump(model, filename='nnmodel4f_v1.2')
