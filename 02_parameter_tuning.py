from dataset import *

### ----------------------------------------------------------------------------
### Learning
### ----------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore") # ignore warnings

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor

### ----------------------------------------------------------------------------
### Grid test
### ----------------------------------------------------------------------------
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html

random_grid = {
    'loss': ['ls', 'lad', 'huber', 'quantile'],
    'n_estimators': [100, 200]
}

model = GradientBoostingRegressor()
random_search = RandomizedSearchCV(
    estimator = model, param_distributions=random_grid, 
    n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_treino, Y_treino)
print(random_search.best_params_)
