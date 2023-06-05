"""
classificação
"""
import matplotlib
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

(previsores_treinamento, previsores_teste), (classe_treinamento, classe_teste) = mnist.load_data()

def vizualizar_n(num_item):
    digit = previsores_treinamento[num_item]
    recost = digit.reshape(28,28)

    plt.imshow(recost, cmap=matplotlib.cm.binary, interpolation='nearest')
    plt.axis('off')
    plt.show()

def qte_itens():
    qte_previsores_treino = previsores_treinamento.shape
    qte_previsores_teste = previsores_teste.shape
    qte_classe_treino = classe_treinamento.shape
    qte_calsse_teste = classe_teste.shape

    print("previsores",qte_previsores_treino, qte_previsores_teste)
    print("respostas", qte_classe_treino, qte_calsse_teste)

def cv_itens(previsores_treinamento=None, previsores_teste=None):
    mistura = np.random.permutation(60000)
    previsores_treinamento, previsores_teste = previsores_treinamento[mistura], previsores_teste[mistura]
    return previsores_treinamento, previsores_teste

previsores_treinamento, previsores_teste = cv_itens(previsores_treinamento, previsores_teste)