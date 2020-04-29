import numpy
import pandas as pd

from dataset import *

### ----------------------------------------------------------------------------
### Learning
### ----------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore") # ignore warnings
from sklearn.ensemble import GradientBoostingRegressor


### ----------------------------------------------------------------------------
### Test
### ----------------------------------------------------------------------------
# grid search by models
from sklearn.metrics import mean_squared_error

## -- Rounding values
def round_for(values, dots=2):
	return numpy.array([round(value,dots) for value in values])

def _in_five(value):
	value = round(value,0)
	u  = value % 10
	ut = 5 if (3 <= u <= 7) else 0
	if (8 <= u):
		value += 10
	value = value - u + ut
	return value

def round_in_five(values):
	return numpy.array([_in_five(value) for value in values])

## -- Prediction
reg = GradientBoostingRegressor(n_estimators=100, loss='ls', random_state=0)
reg.fit(X_treino, Y_treino)
pred = reg.predict(X_teste)

print('normal  ', mean_squared_error(Y_teste, pred))
print('round(2)', mean_squared_error(Y_teste, round_for(pred, 2)))
print('round(0)', mean_squared_error(Y_teste, round_for(pred, 0)))
print('round_5 ', mean_squared_error(Y_teste, round_in_five(pred)))