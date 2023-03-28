"""
validação cruzada
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

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


def criar_rede():
    """
    começo da estrutura neural
    o numero de colunas + o numero de possiveis resopstas > 4 + 3 = 7 / 2 = 3.5
    o softmax retorna o o valor probabilistico de ser umas das opções
    """
    classificador = Sequential()
    # primeira camada oculta
    classificador.add(Dense(units=4, activation='relu', input_dim=4))
    # segunda camada oculta
    classificador.add(Dense(units=4, activation='relu'))
    # neuronios de saida -> 3
    classificador.add(Dense(units=3, activation='softmax'))
    # compilador
    classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return classificador

classificador = KerasClassifier(build_fn=criar_rede, epochs = 1000, batch_size = 10)
"""
o metodo que vamos usar para avaliação atomatica é o evaluete
"""

# resultado
resultados = cross_val_score(estimator=classificador, X=previsores, y=classe_dummy, cv= 10, scoring='accuracy')

media = resultados.mean()
desvio = resultados.std()

print(media)
print(desvio)