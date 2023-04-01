"""
aplicando o modelo
"""
import numpy as np
import pandas as pd
import keras
from keras.models import model_from_json


# para carregar
arquivo = open('./classificador_iris.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('./classificador_iris.h5')

# Criar e classificar novo registro
novo = np.array([[3, 5, 1, 1.2]])
previsao = classificador.predict(novo)
previsao = (previsao > 0.5)
if previsao[0][0] == True and previsao[0][1] == False and previsao[0][2] == False:
    print('Iris setosa')
elif previsao[0][0] == False and previsao[0][1] == True and previsao[0][2] == False:
    print('Iris virginica')
elif previsao[0][0] == False and previsao[0][1] == False and previsao[0][2] == True:
    print('Iris versicolor')

