"""
melhores parametros
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# puxando a base
base = pd.read_csv('../data/iris/iris.csv')

"""
como a base vem com a coluna de resposta temos que separar as caractericas das respostas no caso a classe
"""
# caracteristicas noo caso [todas as linhas, todas as caracteristicas].values
previsores = base.iloc[:, 0:4].values

# coluna com as respostas
classe = base.iloc[:,4].values

# tratando a saida para numeros
labelenconder = LabelEncoder()

# passa a lista para ser encodada no caso transformada em numeros
classe = labelenconder.fit_transform(classe)

# vale lembrar que como serão 3 saidas é preciso transformar em binarios
classe_dummy = np_utils.to_categorical(classe)


def criar_rede(optimizer, loss, activation, neurons):
    """
    começo da estrutura neural
    o numero de colunas + o numero de possiveis resopstas > 4 + 3 = 7 / 2 = 3.5
    o softmax retorna o o valor probabilistico de ser umas das opções
    """
    classificador = Sequential()
    # primeira camada oculta
    classificador.add(Dense(units=neurons, activation=activation, input_dim=4))
    classificador.add(Dropout(0.2))
    # segunda camada oculta
    classificador.add(Dense(units=neurons, activation=activation))
    classificador.add(Dropout(0.2))
    # neuronios de saida -> 3
    classificador.add(Dense(units=3, activation='softmax'))
    # compilador
    classificador.compile(optimizer=optimizer, loss=loss, metrics=['categorical_accuracy'])

    return classificador

classificador = KerasClassifier(build_fn=criar_rede)
"""
o metodo que vamos usar para avaliação atomatica é o evaluete
"""
parametros = {'batch_size':[5,10],
              'epochs':[100, 1000],
              'optimizer':['adam','sgd'],
              'loss':['categorical_crossentropy', 'hinge'],
              'activation':['relu', 'tanh'],
              'neurons':[4,8]}

# chamando o grid para passar o dict com os paramentros
grid_search = GridSearchCV(estimator=classificador, param_grid=parametros, scoring='accuracy', cv=5)

grid_search.fit(previsores, classe)

melhores_prams = grid_search.best_params_
melhor_result = grid_search.best_score_

print("\n##### RNA #####")
print(f"melhores parametros: {melhores_prams}")
print(f"melhores resultados: {melhor_result}")