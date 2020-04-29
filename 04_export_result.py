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

## -- Prediction
reg = GradientBoostingRegressor(n_estimators=100, loss='ls', random_state=0)
reg.fit(X_treino, Y_treino)
pred = reg.predict(X_teste)

print('normal  ', mean_squared_error(Y_teste, pred))
print('round(2)', mean_squared_error(Y_teste, round_for(pred, 2)))
print('round(0)', mean_squared_error(Y_teste, round_for(pred, 0)))

### ----------------------------------------------------------------------------
### Output
### ----------------------------------------------------------------------------
X_desafioqt = dados_desafioqt[coluna_features]
predicao_desafioqt = reg.predict(X_desafioqt)

desafio_df = pd.DataFrame(dados_desafioqt.ID)
desafio_df[coluna_label] = round_for(predicao_desafioqt, 0)

#NÃO TROCAR O NOME DO ARQUIVO DE SAÍDA (PREDICAO_DESAFIO)
desafio_df.to_csv('PREDICAO_DESAFIOQT.csv', index=False) 