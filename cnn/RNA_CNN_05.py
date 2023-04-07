"""
auto melhorando o banco, no caso ele replica a imagem com algumas caracteristicas como leve rotação,
assim melhorando o treinamento caso tenha pouca imagem
"""
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

(X_treino, y_treino), (X_teste, y_teste) = mnist.load_data()

previsores_treinamento = X_treino.reshape(X_treino.shape[0], 28, 28, 1)

previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)

# configura para float, para ter a divisão com decimais
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

# para melhorar vamos dividir por 255 < numero da cor, isso serve para normalizar os dados, assim o maximo é 1
previsores_treinamento /= 255
previsores_teste /= 255

# cria uma matriz com 10 classes no caso é as respostas
classe_treinamento = np_utils.to_categorical(y_treino, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)


classificador = Sequential()

# primeira parte
classificador.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation = 'relu'))
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Flatten())

# rede neural
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dense(units = 10, activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                      metrics = ['accuracy'])

# precisamos de mais imagem de treinamento e não teste
gerador_treinamento = ImageDataGenerator(rotation_range=7, # rotação em graus maximo
                                         horizontal_flip= True, # faz com que a a imagem fique espelhada
                                         shear_range=0.2, # altera o valor dos pixels
                                         height_shift_range= 0.07,  # fazer modificações na altura da imagem
                                         zoom_range= 0.2 # zoom na imagem
                                        )

gerador_teste = ImageDataGenerator()

base_treinamento = gerador_treinamento.flow(previsores_treinamento, classe_treinamento, batch_size=128) # passa os parametros do gerador

base_teste = gerador_teste.flow(previsores_teste, classe_teste, batch_size=128)

# treina com as imagens geradas
classificador.fit(base_treinamento,
                            steps_per_epoch=60000/128,  # como temos 60k de imagem dividimos pelo batch_size
                            epochs=5,
                            validation_data=base_teste,
                            validation_steps=1000/128)

