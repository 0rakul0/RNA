"""
carregando o modelo
"""
import numpy as np
import pandas as pd
import keras
from keras.models import model_from_json


# para carregar
arquivo = open('./classificador_cancer.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('./classificador_cancer.h5')


novoFalso = np.array([[19.81,22.15,130,1260,0.09831,0.1027,0.1479,0.09498,0.1582,0.05395,0.7582,1017,5865,112.4,0.006494,0.01893,0.03391,0.01521,0.01356,0.001997,27.32,30.88,186.8,2398,0.1512,315,0.5372,0.2388,0.2768,0.07615
]])
novoTrue = np.array([[13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,144,0.1773,239,0.1288,0.2977,0.07259
]])

previsaoFalso = classificador.predict(novoFalso)
previsaoFalso = (previsaoFalso > 0.5)
print(previsaoFalso)

previsaoTrue = classificador.predict(novoTrue)
previsaoTrue = (previsaoTrue > 0.5)
print(previsaoTrue)