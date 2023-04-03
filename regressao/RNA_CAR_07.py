"""
melhores parametros
"""
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

base = pd.read_csv('../data/kaggle_car/autos_tratado_sem_valores_null.csv', encoding='ISO-8859-1') # esse encod é pq tem acentos

"""
 0   price                359291 non-null  int64 
 1   abtest               359291 non-null  object 0
 2   vehicleType          359291 non-null  object 1
 3   yearOfRegistration   359291 non-null  int64  2
 4   gearbox              359291 non-null  object 3
 5   powerPS              359291 non-null  int64  4
 6   model                359291 non-null  object 5
 7   kilometer            359291 non-null  int64  6
 8   monthOfRegistration  359291 non-null  int64  7
 9   fuelType             359291 non-null  object 8
 10  brand                359291 non-null  object 9
 11  notRepairedDamage    359291 non-null  object 10
"""

# a coluna 0 é o preço então vamos usar da coluna 1 a 11

previsores = base.iloc[:, 1:13].values # caracteristicas
preco_real = base.iloc[:, 0].values # respostas

"""
o LabelEncoder será usado para transformar texto em numeros para nossa classificação
"""

labelencoder_previsores = LabelEncoder()

# todas as colunas como object listadas do base.info()
previsores[:,0] = labelencoder_previsores.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,10] = labelencoder_previsores.fit_transform(previsores[:,10])

"""
o onehotencoder é usado para os dados não influenciem muito de categoria, os dados que já são numericos são mantidos
"""
# a aplicação do onehotencoder vai deixar como binario ou algo semelhante
ct = ColumnTransformer([('onehotencoder', OneHotEncoder(categories='auto'), [0,1,3,5,8,9,10])], remainder='passthrough') # todas as colunas encodadas

previsores = ct.fit_transform(previsores).toarray()


def criar_rede(loos):  # atualizado: tensorflow==2.0.0-beta1
    regressor = Sequential()
    regressor.add(Dense(units=158, activation='relu', input_dim=317))
    regressor.add(Dense(units=158, activation='relu'))
    regressor.add(Dense(units=1, activation='linear'))
    regressor.compile(loss=loos, optimizer='adam',
                      metrics=['mean_absolute_error'])
    return regressor


# Não é necessário alterar o parâmetro metrics pois ele é usado somente para
# mostrar o resultado e de fato ele não é utilizado no treinamento da rede neural

regressor = KerasRegressor(build_fn = criar_rede, epochs = 100, batch_size = 300)

parametros = {'loos': ['mean_squared_error', 'mean_absolute_error',
                       'mean_absolute_percentage_error', 'mean_squared_logarithmic_error',
                       'squared_hinge']}

grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parametros,
                           cv = 5)
grid_search = grid_search.fit(previsores, preco_real)

melhores_prams = grid_search.best_params_
melhor_result = grid_search.best_score_

print("\n##### RNA #####")
print(f"melhores parametros: {melhores_prams}")
print(f"melhores resultados: {melhor_result}")