"""
criando a rede melhora da rede
"""
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils

# primeira camada Conv2D
# segunda camada MaxPooling2D
# terceira camada Flatten


(X_treino, y_treino), (X_teste, y_teste) = mnist.load_data()


previsores_treinamento = X_treino.reshape(X_treino.shape[0],
                                          28, # largura
                                          28, # altura
                                          1) # canal

previsores_teste = X_teste.reshape(X_teste.shape[0],
                                          28, # largura
                                          28, # altura
                                          1) # canal

# configura para float, para ter a divisão com decimais
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

# para melhorar vamos dividir por 255 < numero da cor, isso serve para normalizar os dados, assim o maximo é 1
previsores_treinamento /= 255
previsores_teste /= 255

# cria uma matriz com 10 classes no caso é as respostas
classe_treinamento = np_utils.to_categorical(y_treino, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

## desenvolvimento da rede
classificador = Sequential()
"""
o primeiro parametro é o 32 filtros, no caso combinações,
o segundo é o tamanho do detector no caso ele é uma matriz 3 por 3
"""
# primeira camada
classificador.add(Conv2D(32, (3,3), input_shape= (28,28,1), activation='relu' ))

# segunda camada
# pool_size -> mapa de caracteristicas
classificador.add(MaxPooling2D(pool_size=(2,2)))

# terceira camada, não precisa de parametros
classificador.add(Flatten())

# rede, add mais uma camada densa com dropout
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=10, activation='softmax'))

classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size=128, epochs=40,
                  validation_data=(previsores_teste, classe_teste))# ja vai fazendo o teste enquanto treina
