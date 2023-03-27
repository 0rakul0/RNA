"""
ajustando os pesos
"""
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

def criar_rede(optimizer, loss, kernel_initializer, activation, neurons):
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
    return classificador

# chama a rede
classificador = KerasClassifier(build_fn=criar_rede)

# paramentros de configuração
parametros = {'batch_size':[10,30],
              'epochs':[50, 100],
              'optimizer':['adam','sgd'],
              'loss':['binary_crossentropy', 'hinge'],
              'kernel_initializer':['random_uniform','normal'],
              'activation':['relu', 'tanh'],
              'neurons':[16,8]}

# chamando o grid para passar o dict com os paramentros
grid_search = GridSearchCV(estimator=classificador, param_grid=parametros, scoring='accuracy', cv=5)

grid_search.fit(previsores, classe)

melhores_prams = grid_search.best_params_
melhor_result = grid_search.best_score_

print("\n##### RNA #####")
print(f"melhores parametros: {melhores_prams}")
print(f"melhores resultados: {melhor_result}")
