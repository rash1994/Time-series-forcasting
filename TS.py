# fit an AR model and manually save coefficients to file
from pandas import read_csv
from statsmodels.tsa.ar_model import AutoReg
import numpy
import csv
d1 = 1
d2 = 30

def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)

def predict(coef, history):
	yhat = coef[0]
	for i in range(1, len(coef)):
		yhat += coef[i] * history[-i]
	return yhat

series = read_csv('DATA3.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
X = difference(series.values)
window_size = 6
model = AutoReg(X, lags=window_size)
model_fit = model.fit()
coef = model_fit.params
numpy.save('man_model.npy', coef)
lag = X[-window_size:]
numpy.save('man_data.npy', lag)
numpy.save('man_obs.npy', [series.values[-1]])
coef = numpy.load('man_model.npy')
print(coef)
lag = numpy.load('man_data.npy')
print(lag)
last_ob = numpy.load('man_obs.npy')
print(last_ob)

with open('results.csv', 'w') as csvfile:
	writer = csv.writer(csvfile)

for x in range(30):
	coef = numpy.load('man_model.npy')
	lag = numpy.load('man_data.npy')
	last_ob = numpy.load('man_obs.npy')
	prediction = predict(coef, lag)
	yhat = prediction + last_ob[0]
	writer.writerow(yhat)
	print('Prediction on day: %f' % yhat)
	observation = yhat
	diffed = observation - last_ob[0]
	lag = numpy.append(lag[1:], [diffed], axis=0)
	numpy.save('man_data.npy', lag)
	last_ob[0] = observation
	numpy.save('man_obs.npy', last_ob)

	x=x+1





# val = input("Enter the Date: ")
# val=int(val)
# print(val)

#
# if 1 < val < 30:
# 	print('data')
# else:
# 	print('No!')
