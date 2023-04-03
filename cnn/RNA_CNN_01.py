"""
nesse tipo de cnn vamos monstar a base para fazer o pre-processamento meio que automatico
para depois aplicar a rede
"""
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils

# primeira camada Conv2D
# segunda camada MaxPooling2D
# terceira camada Flatten


(X_treino, y_treino), (X_teste, y_teste) = mnist.load_data()
plt.imshow(X_treino[4], cmap='gray') #deixa no canal de cinza
plt.title('Classe'+ str(y_treino[4]))
plt.show()

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

