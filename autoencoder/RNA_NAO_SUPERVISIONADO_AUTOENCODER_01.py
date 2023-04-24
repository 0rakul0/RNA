"""
preparação do autoencoder
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense

# carregamentos da base
(previsores_treinamento, _), (previsores_teste, _) = mnist.load_data()

#  normaliza
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

# da os 784 pixel que seria 28 * 28
previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:])))
previsores_teste = previsores_teste.reshape((len(previsores_teste), np.prod(previsores_teste.shape[1:])))

# camada de imput 784 - 32 - 784
fator_compactacao = 784 / 32

autoencoder = Sequential()
autoencoder.add(Dense(units=32, activation='relu', input_dim=784))
autoencoder.add(Dense(units=784, activation='sigmoid'))

# vizualizaçao da rede
print(autoencoder.summary())
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 32)                25120     
_________________________________________________________________
dense_1 (Dense)              (None, 784)               25872     
=================================================================
Total params: 50,992
Trainable params: 50,992
Non-trainable params: 0
_________________________________________________________________
"""
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# vamos passar a entrada e saida sendo o mesmo treinamento
autoencoder.fit(previsores_treinamento, previsores_treinamento, epochs=100, batch_size=256,
                validation_data=(previsores_teste, previsores_teste))