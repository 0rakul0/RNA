"""
ativardores
"""

import numpy as np

stepFunction = lambda soma: 1 if soma >= 1 else 0 # ou é 1 ou é 0

relu = lambda soma: soma if soma >= 0 else 0 # do 0 ao maximo sem parar... tomar cuidado pois não tem limite

linear = lambda soma: soma # retorna o proprio valor

sigmoid = lambda soma: 1 / (1 + np.exp(-soma)) # valores que ficam entre -1 e 1

tangHyperbolica = lambda soma: (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma)) # valores que ficam entre -1 e 1 melhorada

softmax = lambda soma: np.exp(soma) / np.sum(np.exp(soma)) # é usada para prever a possibilidade de outras classes


testeSoma = -0.99
teste = stepFunction(testeSoma)
testeS = sigmoid(testeSoma)
testeT = tangHyperbolica(testeSoma)
testeR = relu(testeSoma)
testeL = linear(testeSoma)
testeSF = softmax([5, 2, 1.3, 4])

print(teste,'\n')
print(testeS,'\n')
print(testeT,'\n')
print(testeR,'\n')
print(testeL,'\n')
print(testeSF,'\n')