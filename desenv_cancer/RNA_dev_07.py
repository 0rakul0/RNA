"""
aplicando os pesos e testando o modelo
"""
import numpy as np
import pandas as pd
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# importa os dados já tratados
previsores = pd.read_csv('../data/breast_cancer/entradas_breast.csv')
classe = pd.read_csv('../data/breast_cancer/saidas_breast.csv')

# melhores indices
optimizer='adam'
loss='binary_crossentropy'
kernel_initializer='random_uniform'
activation='relu'
neurons=16

# chama o sequenciador
classificador = Sequential()
"""
claculo para saber quantos neuronios usar para a camanda oculta basta somar entrada e saida e dividir por dois
entrada é o numero de colunas do dataset, no caso atual é 30
e para saida é 1

logo usamos 30+1 / 2 = 15.5 arredonda para cima = 16

para compilar o modelo, basta chamar o compile e para treinar usar o fit passando a base e juste dos pesos
"""
# primeira camada oculta, já deixando conectada
classificador.add(Dense(units=neurons,  # quantidade de neuronios nas
                        activation=activation,  # ativador
                        kernel_initializer=kernel_initializer,  # aleatoriedade
                        input_dim=30  # quantidade de colunas de entrada
                        ))
classificador.add(Dropout(0.2)) # da um pequeno apagão em alguns neuronios para melhor aprendizado
# segunda camada oculta
classificador.add(Dense(units=neurons,  # quantidade de neuronios nas
                        activation=activation,  # ativador
                        kernel_initializer=kernel_initializer,  # aleatoriedade
                        ))
classificador.add(Dropout(0.2))  # da um pequeno apagão em alguns neuronios para melhor aprendizado
# camada de saida
classificador.add(Dense(units=1,  # proximo de zero é cancer benigno, proximo de 1 é cancer ruim
                        activation='sigmoid'
                        ))
# compilador
classificador.compile(optimizer=optimizer,
                      loss=loss, # usado para classificadores binarios
                      metrics= ['binary_accuracy']) # como a resposta é binaria é recomendado usar esse)

classificador.fit(previsores, classe, batch_size=10,epochs=100)

novo = np.array([[12.45,15.7,82.57,477.1,0.1278,0.17,0.1578,0.08089,0.2087,0.07613,0.3345,0.8902,2217,27.19,
                  0.00751,0.03345,0.03672,0.01137,0.02165,0.005082,15.47,23.75,103.4,741.6,0.1791,0.5249,
                  0.5355,0.1741,0.3985,0.1244]])

previsao = classificador.predict(novo)
previsao = previsao>0.5
print(previsao)
