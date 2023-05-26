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

# vizualizando
dimensao_original = Input(shape=(784,))
camada_encoder = autoencoder.layers[0]
encoder = Model(dimensao_original, camada_encoder(dimensao_original))
print(encoder.summary())

"""
para aplicações comerciais precisa salvar ambos os autoencoder como o encoder
"""
imagens_codiifcadas = encoder.predict(previsores_teste)
imagens_decodificadas = autoencoder.predict(previsores_teste)

# para ver os resultados
numero_imagens = 10
imagens_teste = np.random.randint(previsores_teste.shape[0], size = numero_imagens)

"""
aqui vamos mostrar 3 etapas, 
a primeira é a imagem original,
a segunda é a imagem codificada,
a terceira é a imagem reconstruida
"""

plt.figure(figsize=(18,18))
for i, indice_imagem in enumerate(imagens_teste):
    print(i, indice_imagem)
    # imagem original
    eixo = plt.subplot(10, 10, i+1)
    plt.imshow(previsores_teste[indice_imagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())

    # imagem codificada
    eixo = plt.subplot(10, 10, i + 1 + numero_imagens)
    plt.imshow(imagens_codiifcadas[indice_imagem].reshape(8, 4)) # 8 * 4 = 32 // linha 23
    plt.xticks(())
    plt.yticks(())

    # imagem decodificada
    eixo = plt.subplot(10, 10, i + 1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())

plt.savefig('../data/img/aprendizado_nao_supervisionado.png', dpi=300)
plt.show()