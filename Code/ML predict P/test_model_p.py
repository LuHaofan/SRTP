# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:12:55 2020

@author: lhf
"""

from sklearn.externals import joblib
import csv
import matplotlib.pyplot as plt
import numpy as np

#%%load model and scaler
modelname = 'nnmodel4f_v1.2'
nn_model = joblib.load(modelname)
scalername = '4fscaler'
scaler = joblib.load(scalername)
pmodelname = 'nnmodelFP_v1.1'
p_model = joblib.load(pmodelname)
pscalername = 'FindPscaler'
p_scaler = joblib.load(pscalername)


#%%define the funciton to compute the error
def costFunction(y_pred, y):
    cost = 0
    for j in range(len(y)):
        cost += (y_pred[j]-y[j])**2
    return (1/2/len(y))*cost

#%% predict p value
fname = 'test_file.csv'
print('Loading data ...\n')
kx = []
ky = []
X = []
y = []
with open(fname) as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for row in csv_reader:
        kx.append(row[0:3])
        ky.append(row[7])
        X.append(row[3:6])
        y.append(row[6])
        
kx = np.array([[float(i) for i in row] for row in kx])
ky = np.array([float(j) for j in ky])
X = [[float(i) for i in row] for row in X]
y = [float(j) for j in y]
px = np.array(X[:])
px = p_scaler.transform(px)
pavg = []
for m in range(10):
    py = p_model[m].predict(px)
    pavg.append(list(py))
pavg = np.array(pavg)
pavg = np.mean(pavg,0)
pavg = np.expand_dims(pavg, axis = 1)
#print(np.shape(kx))
#print(np.shape(pavg))
kx = np.hstack((kx, pavg))
#print(kx)
for i in range(len(pavg)):
    print('{}:{}'.format(i+2, pavg[i]))
#%% test the model
Test_X = kx
Test_y = ky
Test_X = scaler.transform(Test_X)

kavg = []
for m in range(10):
    pred_y = nn_model[m].predict(Test_X)
    kavg.append(list(pred_y))
kavg = np.array(kavg)
pred = np.mean(kavg,0)

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
    print('\t{} \t{} \t{} \t{} '.format(i+2, pred[i], Test_y[i], Test_error[i]))
print('\n')
print('Average Cost:')
print(costFunction(pred, Test_y))

plt.figure()
plt.plot(np.linspace(1,100,100), Test_error)
plt.title('NN test error plot')
plt.xlabel('Index of test samples')
plt.ylabel('relative error %')
plt.show()




