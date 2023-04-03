"""
rede neural
"""
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

base = pd.read_csv('../data/videogame/games.csv')

"""
base.info()
0   Name             16717 non-null  object # transformar
1   Platform         16719 non-null  object # transformar
2   Year_of_Release  16450 non-null  float64
3   Genre            16717 non-null  object # transformar
4   Publisher        16665 non-null  object # transformar
5   NA_Sales         16719 non-null  float64
6   EU_Sales         16719 non-null  float64
7   JP_Sales         16719 non-null  float64
8   Other_Sales      16719 non-null  float64
9   Global_Sales     16719 non-null  float64
10  Critic_Score     8137 non-null   float64
11  Critic_Count     8137 non-null   float64
12  User_Score       10015 non-null  object # transformar
13  User_Count       7590 non-null   float64
14  Developer        10096 non-null  object # transformar
15  Rating           9950 non-null   object # transformar
"""

# limpar base
base = base.drop(['Other_Sales','Developer'], axis=1)

"""
base.info()
 0   Name             16717 non-null  object 
 1   Platform         16719 non-null  object 
 2   Year_of_Release  16450 non-null  float64
 3   Genre            16717 non-null  object 
 4   Publisher        16665 non-null  object 
 5   NA_Sales         16719 non-null  float64
 6   EU_Sales         16719 non-null  float64
 7   JP_Sales         16719 non-null  float64
 8   Global_Sales     16719 non-null  float64
 9   Critic_Score     8137 non-null   float64
 10  Critic_Count     8137 non-null   float64
 11  User_Score       10015 non-null  object 
 12  User_Count       7590 non-null   float64
 13  Rating           9950 non-null   object 
"""

# tirando os NAN
base = base.dropna(axis=0) # apagas todas as linhas que tiver nan

# verificando os nomes
"""
print(base['Name'].value_counts())
Length: 223 nomes diferentes
"""
# como o nome é muito importante vamos deixar de fora por enquanto
nome_jogos = base.Name
base = base.drop('Name', axis=1)

# eleminando as clunas que não vamos usar
base = base.drop(['NA_Sales','EU_Sales','JP_Sales'], axis =1)

base = base.loc[base['Global_Sales'] > 1]

"""
base.info()
Data columns (total 12 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   Platform         6825 non-null   object 
 1   Year_of_Release  6825 non-null   float64
 2   Genre            6825 non-null   object 
 3   Publisher        6825 non-null   object 
 4   Global_Sales     6825 non-null   float64 
 5   Critic_Score     6825 non-null   float64
 6   Critic_Count     6825 non-null   float64
 7   User_Score       6825 non-null   object 
 8   User_Count       6825 non-null   float64
 9   Rating           6825 non-null   object 
dtypes: float64(7), object(5)
"""

previsores = base.iloc[:,[0,1,2,3,5,6,7,8,9]].values
venda_global = base.iloc[:,4].values

"""print(previsores[0])"""
# transformar todos que são texto em numero
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
previsores[:,8] = labelencoder.fit_transform(previsores[:,8])
"""print(previsores[0])"""

# aplicando o onehotencoder
ct = ColumnTransformer([('onehotencoder', OneHotEncoder(categories='auto'), [0,2,3,8])], remainder='passthrough') # todas as colunas encodadas

previsores = ct.fit_transform(previsores).toarray()

"""
a quantidade de colunas geradas pelo onehotencoder foi de 61
61+3 / 2 = 32
"""
# inicio da rede estrutura
ativador = Activation(activation='sigmoid')
camada_entrada = Input(shape=99)
camada_oculta1 = Dense(units=50, activation=ativador)(camada_entrada)
drops1 = Dropout(0.2)(camada_oculta1)
camada_oculta2 = Dense(units=50, activation=ativador)(drops1)
drops2 = Dropout(0.2)(camada_oculta2)
camada_oculta3 = Dense(units=50, activation=ativador)(drops2)
camada_saida_global = Dense(units=1, activation='linear')(camada_oculta3)



regressor = Model(inputs= camada_entrada, outputs= camada_saida_global)
regressor.compile(optimizer='adam', loss='mse')
regressor.fit(previsores, venda_global, epochs=6000, batch_size=100)
previsao_global = regressor.predict(previsores)