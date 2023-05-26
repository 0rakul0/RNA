"""
classificação usando autoencoder
"""

import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.utils import np_utils

# carregamentos da base
(previsores_treinamento, classe_treinamento), (previsores_teste, classe_teste) = mnist.load_data()
#  normaliza
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

# gerando as colunas de respostas ficando um marcador para na coluna resposta
classe_dummy_treinamento = np_utils.to_categorical(classe_treinamento)
classe_dummy_teste = np_utils.to_categorical(classe_teste)

# da os 784 pixel que seria 28 * 28
previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:])))
previsores_teste = previsores_teste.reshape((len(previsores_teste), np.prod(previsores_teste.shape[1:])))

autoencoder = Sequential()
autoencoder.add(Dense(units = 32, activation = 'relu', input_dim = 784))
autoencoder.add(Dense(units = 784, activation = 'sigmoid'))
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                    metrics = ['accuracy'])
autoencoder.fit(previsores_treinamento, previsores_treinamento,
                epochs = 100, batch_size = 256,
                validation_data = (previsores_teste, previsores_teste))

# vizualizando
dimensao_original = Input(shape=(784,))
camada_encoder = autoencoder.layers[0]
encoder = Model(dimensao_original, camada_encoder(dimensao_original))

# aplica o encoder
previsores_treinamento_codificados = encoder.predict(previsores_treinamento)
previsores_teste_codificados = encoder.predict(previsores_teste)

# modelo sem redução de dimencionalidade
c1 = Sequential()
# 784 + 10 = 794 / 2 = 397
c1.add(Dense(units=397, activation='relu', input_dim = 784))
c1.add(Dense(units=397, activation='relu'))
c1.add(Dense(units=10, activation='softmax'))
c1.compile(optimizer = 'adam', loss='categorical_crossentropy',
           metrics = ['accuracy'])
c1.fit(previsores_treinamento, classe_dummy_treinamento, batch_size = 256,
       epochs = 100, validation_data=(previsores_teste, classe_dummy_teste))

# com redução de dimensionalidade
c2 = Sequential()
c2.add(Dense(units = 21, activation = 'relu', input_dim = 32))
c2.add(Dense(units = 21, activation = 'relu'))
c2.add(Dense(units = 10, activation = 'softmax'))
c2.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
           metrics = ['accuracy'])
c2.fit(previsores_treinamento_codificados, classe_dummy_treinamento, batch_size = 256,
       epochs = 100, validation_data=(previsores_teste_codificados, classe_dummy_teste))

print(c1.summary())

"""
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_2 (Dense)              (None, 397)               311645    
_________________________________________________________________
dense_3 (Dense)              (None, 397)               158006    
_________________________________________________________________
dense_4 (Dense)              (None, 10)                3980      
=================================================================
Total params: 473,631
Trainable params: 473,631
Non-trainable params: 0
_________________________________________________________________
"""
print(c2.summary())
"""
None
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_5 (Dense)              (None, 21)                693       
_________________________________________________________________
dense_6 (Dense)              (None, 21)                462       
_________________________________________________________________
dense_7 (Dense)              (None, 10)                220       
=================================================================
Total params: 1,375
Trainable params: 1,375
Non-trainable params: 0
_________________________________________________________________
None
"""