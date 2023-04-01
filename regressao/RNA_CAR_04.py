"""
OneHotEncoder
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
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
print(previsores)