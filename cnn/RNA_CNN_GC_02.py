"""
CNN para gatos e cachorros
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

classificador = Sequential()

# input das imagens vindo de uma matriz
classificador.add(Conv2D(32, (3,3),
                         input_shape=(64, 64, 3), # dimessões e canais, 64 x 64 e RGB
                         activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2))) # faz uma matriz 2 x 2 e vai passando para pegar o maior valor
classificador.add(Flatten())

# rede neural
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
# classificar somente uma imagem
classificador.add(Dense(units=1, activation='sigmoid'))

classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

gerador_treino = ImageDataGenerator(rescale=1./255, # ajuda na normalização
                                    rotation_range=7,
                                    horizontal_flip=True,
                                    shear_range= 0.2,
                                    height_shift_range=0.7,
                                    zoom_range=0.2
                                    )
gerador_teste = ImageDataGenerator(rescale=1./255)

base_treino = gerador_treino.flow_from_directory('../data/gatos_cachorros/treino',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')
base_teste = gerador_teste.flow_from_directory('../data/gatos_cachorros/teste',
                                               target_size=(64, 64),
                                               batch_size=32,
                                               class_mode='binary')

classificador.fit(base_treino,
                  steps_per_epoch=4000/32, # quatidade de imagens de treino passando uma a uma
                  epochs=10,
                  validation_data= base_teste,
                  validation_steps=1000/32 # quatidade de imagens de teste passando uma a uma
                  )

imagem_teste = image.load_img('../data/gatos_cachorros/treino/gato/cat.0.jpg', target_size=(64,64))

imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste, axis=0)

previsao = classificador.predict(imagem_teste)
print(base_treino.class_indices)

previsao = (previsao > 0.5)
if previsao == True:
    print('cachorro')
elif previsao == False:
    print('gato')

