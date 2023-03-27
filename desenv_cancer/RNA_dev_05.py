"""
aplicando dropout
"""
import pandas as pd
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# importa os dados já tratados
previsores = pd.read_csv('../data/breast_cancer/entradas_breast.csv')
classe = pd.read_csv('../data/breast_cancer/saidas_breast.csv')

def criar_rede():
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
    classificador.add(Dense(units=16,  # quantidade de neuronios nas
                            activation='relu',  # ativador
                            kernel_initializer='random_uniform',  # aleatoriedade
                            input_dim=30  # quantidade de colunas de entrada
                            ))
    classificador.add(Dropout(0.2)) # da um pequeno apagão em alguns neuronios para melhor aprendizado
    # segunda camada oculta
    classificador.add(Dense(units=16,  # quantidade de neuronios nas
                            activation='relu',  # ativador
                            kernel_initializer='random_uniform',  # aleatoriedade
                            ))
    classificador.add(Dropout(0.2))  # da um pequeno apagão em alguns neuronios para melhor aprendizado
    # camada de saida
    classificador.add(Dense(units=1,  # proximo de zero é cancer benigno, proximo de 1 é cancer ruim
                            activation='sigmoid'
                            ))

    otimizador = keras.optimizers.Adam(learning_rate=0.001, decay=0.001, clipvalue=0.5)

    # compilador
    classificador.compile(optimizer=otimizador,
                          loss='binary_crossentropy', # usado para classificadores binarios
                          metrics= ['binary_accuracy']) # como a resposta é binaria é recomendado usar esse)
    return classificador


classificador = KerasClassifier(build_fn = criar_rede, # chama a função para criar a rede
                                epochs = 100, batch_size = 10)

resultados = cross_val_score(estimator=classificador, # pega a rede classificada
                             X=previsores, y=classe, # faz a validação cruzada
                             cv=10, scoring='accuracy')


print('array com os resultados: ', resultados)

# como é gerado um array com 10 valores, usamo então a média deles
media_resultado = resultados.mean()
desvio_padrao = resultados.std()

print('media: ', media_resultado)
print('desvio_padrao', desvio_padrao)