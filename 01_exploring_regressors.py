import numpy
import pandas as pd

from dataset import *

### ----------------------------------------------------------------------------
### Learning
### ----------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore") # ignore warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import HuberRegressor

models = [
    RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor,
    GradientBoostingRegressor, AdaBoostRegressor, HuberRegressor
]

### ----------------------------------------------------------------------------
### Test
### ----------------------------------------------------------------------------
# grid search by models
from sklearn.metrics import mean_squared_error

for model in models:
    reg = model()
    reg.fit(X_treino, Y_treino)
    pred = reg.predict(X_teste)
    evaluation = mean_squared_error(Y_teste, pred)
    print(model, evaluation)