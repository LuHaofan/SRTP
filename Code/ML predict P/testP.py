# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:03:50 2020

@author: lhf
"""

from sklearn.externals import joblib
import csv
import matplotlib.pyplot as plt
import numpy as np
#%% define the cost function
def costFunction(y_pred, y):
    cost = 0
    for j in range(len(y)):
        cost += (y_pred[j]-y[j])**2
    return (1/2/len(y))*cost
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

avg = []
for m in range(10):
    pred_y = nn_model[m].predict(Test_X)
    avg.append(list(pred_y))
    
avg = np.array(avg)
pred = np.mean(avg,0)

#pred = nn_model.predict(Test_X)
Test_error = []
for i in range(len(Test_y)):
    Test_error.append(100*abs(pred[i]-Test_y[i])/Test_y[i])

temp = Test_error[:]
temp.sort()
temp.reverse()
for i in range(5):
    print(i+1)
    print('error:  %f\n' %temp[i])
    print('index:  %d\n' %Test_error.index(temp[i]))
    print(X[Test_error.index(temp[i])])
    print('\n')

print('\tidx \tpred_y \tTest_y \terror')
for i in range(100):
    print('\t{} \t{} \t{} \t{}'.format(i+2, pred[i], Test_y[i], Test_error[i]))
print('\n')
print('Average Cost:')
print(costFunction(pred, Test_y))
print('Mean error:{}'.format(np.mean(Test_error)))
print('Standard deviation Error:{}'.format(np.std(Test_error)))
plt.figure()
plt.plot(np.linspace(1,100,100), Test_error)
plt.title('NN test error plot')
plt.xlabel('Index of test samples')
plt.ylabel('relative error %')
plt.show()

