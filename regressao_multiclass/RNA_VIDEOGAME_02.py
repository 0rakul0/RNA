"""
Montando os previsores
"""
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

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
base = base.drop(['Other_Sales','Global_Sales','Developer'], axis=1)
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
 8   Critic_Score     8137 non-null   float64
 9   Critic_Count     8137 non-null   float64
 10  User_Score       10015 non-null  object 
 11  User_Count       7590 non-null   float64
 12  Rating           9950 non-null   object 
"""

# tirando os NAN
base = base.dropna(axis=0) # apagas todas as linhas que tiver nan

"""
print(base.shape)
(6825, 13)
"""
# localizando outlies
america_menor_q_1 = base.loc[base['NA_Sales']<1]
"""print(america_menor_q_1)"""
base = base.loc[base['NA_Sales']>1]
base = base.loc[base['EU_Sales']>1]
"""print(base.shape) #(258, 13)"""

# verificando os nomes
"""
print(base['Name'].value_counts())
Length: 223 nomes diferentes
"""
# como o nome é muito importante vamos deixar de fora por enquanto
nome_jogos = base.Name
base = base.drop('Name', axis=1)

"""base.info()
Data columns (total 12 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   Platform         258 non-null    object    sim
 1   Year_of_Release  258 non-null    float64   sim
 2   Genre            258 non-null    object    sim
 3   Publisher        258 non-null    object    sim
 4   NA_Sales         258 non-null    float64   classe
 5   EU_Sales         258 non-null    float64   classe
 6   JP_Sales         258 non-null    float64   classe
 7   Critic_Score     258 non-null    float64   sim
 8   Critic_Count     258 non-null    float64   sim
 9   User_Score       258 non-null    object    sim
 10  User_Count       258 non-null    float64   sim
 11  Rating           258 non-null    object    sim
dtypes: float64(7), object(5)
"""

previsores = base.iloc[:,[0,1,2,3,7,8,9,10,11]].values
venda_na = base.iloc[:,4].values
venda_ue = base.iloc[:,5].values
venda_jp = base.iloc[:,6].values