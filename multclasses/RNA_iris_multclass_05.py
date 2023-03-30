"""
salvando o modelo
melhores parametros: {
'activation': 'relu',
'batch_size': 30,
'dropout': 0.4,
'epochs': 3000,
'kernel_initializer': 'normal',
'loss': 'categorical_crossentropy',
'neurons': 16,
'optimizer': 'sgd'}
melhores resultados: 0.9666666666666666
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
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
# tratando a saida para numeros
labelenconder = LabelEncoder()
# passa a lista para ser encodada no caso transformada em numeros
classe = labelenconder.fit_transform(classe)
# vale lembrar que como serão 3 saidas é preciso transformar em binarios
classe_dummy = np_utils.to_categorical(classe)

previsores_treino, previsores_teste, classe_treino, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

classificador = Sequential()

# primeira camada oculta
classificador.add(Dense(units=16, activation='relu',kernel_initializer = 'normal', input_dim=4))
classificador.add(Dropout(0.4))
# segunda camada oculta
classificador.add(Dense(units=16, activation='relu', kernel_initializer = 'normal'))
classificador.add(Dropout(0.4))
# neuronios de saida -> 3
classificador.add(Dense(units=3, activation='softmax'))
# compilador
classificador.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# treinamento
classificador.fit(previsores_treino, classe_treino, batch_size=30, epochs=3000)
"""
o metodo que vamos usar para avaliação atomatica é o evaluete
"""
classificador_json = classificador.to_json()
with open('classificador_iris.json','w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_iris.h5')
