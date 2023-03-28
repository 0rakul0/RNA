"""
montando e tratando base
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# puxando a base
base = pd.read_csv('../data/iris/iris.csv')

"""
como a base vem com a coluna de resposta temos que separar as caractericas das respostas no caso a classe
"""
# caracteristicas noo caso [todas as linhas, todas as caracteristicas].values
previsores = base.iloc[:, 0:4].values

# coluna com as respostas
classe = base.iloc[:,4].values

# tratando a saida para numeros
labelenconder = LabelEncoder()

# passa a lista para ser encodada no caso transformada em numeros
classe = labelenconder.fit_transform(classe)

# vale lembrar que como serão 3 saidas é preciso transformar em binarios
classe_dummy = np_utils.to_categorical(classe)

"""
uma vez já separada precisamos montar quem é treino e quem é teste, usando o sklearn
"""
# separa em treino e teste
previsores_treino, previsores_teste, classe_treino, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

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
# compilador
classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# treinamento
classificador.fit(previsores_treino, classe_treino, batch_size=10, epochs=1000)

"""
o metodo que vamos usar para avaliação atomatica é o evaluete
"""
# resultado
resultado = classificador.evaluate(previsores_teste, classe_teste)

# previsões
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

# como a matriz de confusão só aceita numero de uma coluna aqui estou convertendo novamente a versão original
classe_teste_np = [np.argmax(t) for t in classe_teste]
previsoes_np = [np.argmax(t) for t in previsoes]


# matrix de confusão
matriz = confusion_matrix(previsoes_np, classe_teste_np)
print(matriz)
