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

for i in range(0, 100):
    reg = GradientBoostingRegressor(n_estimators=100, loss='ls', random_state=i)
    reg.fit(X_treino, Y_treino)
    pred = reg.predict(X_teste)
    evaluation = mean_squared_error(Y_teste, pred)
    print(i, evaluation)