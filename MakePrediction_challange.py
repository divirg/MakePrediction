#!/usr/bin/python
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import os

print(os.getcwd())
#read data
x_values=pd.read_csv('challenge_dataset.txt', usecols = [0], header=None)
y_values=pd.read_csv('challenge_dataset.txt', usecols = [1], header=None)


#train model on data
body_reg=linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#make prediction
prediction= body_reg.predict(x_values)

sample=12
y_sample=y_values.iloc[sample]
x_sample=x_values.iloc[sample]
prediction_sample=prediction[sample]
error=prediction_sample-y_sample

print('dataset X=%f, Y=%f - prediction=%f - error=%f'  % (x_sample,  y_sample, prediction_sample,  error))

#plot results
plt.scatter(x_values, y_values)
plt.scatter(x_sample,y_sample,color='red', s=95)
plt.scatter(x_sample,prediction_sample,color='green', s=75)
plt.plot(x_values, prediction)
plt.text(0, 25, 'dataset X=%f, Y=%f - prediction=%f - error=%f'  % (x_sample,  y_sample, prediction[28],  error))
plt.show()
