#!/usr/bin/python
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import os
import numpy as np

print(os.getcwd())

#read data
df = pd.read_csv('challenge_dataset_3.csv', header=None,  names=['X1', 'X2', 'X3', 'Y'])
x_values=df[['X1', 'X2', 'X3']]
y_values=df['Y']
x1=x_values['X1']
x2=x_values['X2']
x3=x_values['X3']

#train model on data
body_reg=linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#make prediction
prediction= body_reg.predict(x_values)

plt.figure(1)
ax1=plt.subplot(211)
plt.scatter(x1, x2, x3, y_values)
plt.subplot(212, sharex=ax1)
plt.scatter(x1, x2, x3, prediction)
plt.show()

result=[]

for k in range(y_values.count()) :
	sample=k
	y_sample=y_values.iloc[sample]
	x1_sample=x1.iloc[sample]
	x2_sample=x2.iloc[sample]
	x3_sample=x3.iloc[sample]
	prediction_sample=prediction[sample]
	error=prediction_sample-y_sample
	result.append(abs(error))
	print('error=%f - prediction=%f - dataset X=%f,%f%f Y=%f '  \
		% (error, prediction_sample,  x1_sample, x2_sample, x3_sample,  y_sample))

print("Max %f" % (max(result)))
print("Min %f" % (min(result)))
print("Average %f" % (np.mean(result)))

##plot results
#plt.scatter(x_values, y_values)
#plt.scatter(x_sample,y_sample,color='red', s=95)
#plt.scatter(x_sample,prediction_sample,color='green', s=75)
#plt.plot(x_values, prediction)
#plt.text(0, 25, 'dataset X=%f, Y=%f - prediction=%f - error=%f'  % (x_sample,  y_sample, prediction[28],  error))
#plt.show()

