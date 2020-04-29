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

reg = GradientBoostingRegressor(n_estimators=100, loss='ls', random_state=0)
reg.fit(X_treino, Y_treino)
pred = reg.predict(X_teste)
evaluation = mean_squared_error(Y_teste, pred)
print(GradientBoostingRegressor, evaluation)

### ----------------------------------------------------------------------------
### Output
### ----------------------------------------------------------------------------
X_desafioqt = dados_desafioqt[coluna_features]
predicao_desafioqt = reg.predict(X_desafioqt)

desafio_df = pd.DataFrame(dados_desafioqt.ID)
desafio_df[coluna_label] = predicao_desafioqt

#NÃO TROCAR O NOME DO ARQUIVO DE SAÍDA (PREDICAO_DESAFIO)
desafio_df.to_csv('PREDICAO_DESAFIOQT.csv', index=False) 