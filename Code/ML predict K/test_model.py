from sklearn.externals import joblib
import csv
import matplotlib.pyplot as plt
import numpy as np

#%%load model and scaler
modelname = '../../Model/FindKnnmodel4f_v1'
nn_model = joblib.load(modelname)
scalername = '../../Model/FindK4fscaler'
scaler = joblib.load(scalername)

#%%define the funciton to compute the error
def costFunction(y_pred, y):
    cost = 0
    for j in range(len(y)):
        cost += (y_pred[j]-y[j])**2
    return (1/2/len(y))*cost

#%%load the test data
fname = '../../Data/randEllipsoid_4f_test.csv'
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

    
#%% Test Ellipsoid
a = 0.49
b = 0.3
c = 0.3
V = (4/3)*np.pi*a*b*c
f = V
p = c/a
Kp = [0.2, 0.5, 1, 2, 5, 10, 20]
Km = Kp
filename = 'ellipsoid.csv'
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




