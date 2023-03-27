"""
montando a base
"""
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


# puxando a base
base = pd.read_csv('../data/iris/iris.csv')

"""
como a base vem com a coluna de resposta temos que separar as caractericas das respostas no caso a classe
"""
# caracteristicas noo caso [todas as linhas, todas as caracteristicas].values
previsores = base.iloc[:, 0:4].values

# coluna com as respostas
classe = base.iloc[:,4].values

"""
uma vez já separada precisamos montar quem é treino e quem é teste, usando o sklearn
"""

# separa em treino e teste
previsores_treino, previsores_teste, classe_treino, classe_teste = train_test_split(previsores, classe, test_size=0.25)

"""
começo da estrutura neural
o numero de colunas + o numero de possiveis resopstas > 4 + 3 = 7 / 2 = 3.5
o softmax retorna o o valor probabilistico de ser umas das opções
"""
classificador = Sequential()

# primeira camada oculta
classificador.add(Dense(units=4, activation='relu', input_dim=4))
# segunda camada oculta
classificador.add(Dense(units=4, activation='relu'))
# neuronios de saida -> 3
classificador.add(Dense(units=3, activation='softmax'))

