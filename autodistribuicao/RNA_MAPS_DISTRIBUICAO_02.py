"""
o objetivo é ver quem pode ser uma possivel fraude

"""
import numpy as np
from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot
import matplotlib.pyplot as plt
import math

base = pd.read_csv('../data/maps_auto_rganizaveis/credit_data.csv')

# limpeza da base apos uma analize de como ela está
base = base.dropna()
base.loc[base.age < 0, 'age'] = 40.92

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:,4].values

# normalizar os dados
normalizador = MinMaxScaler(feature_range=(0,1))

previsores = normalizador.fit_transform(previsores)

# criação do mapa
"""
para saber o x e y - pega o numero de linhas, tira a raiz quadrada e multiplica por 5. 
e tira a raiz quadrada novamente
x = linhas horizontais
y = linhas verticais
input_len = numero de caracteristicas no caso numero de colunas
sigma é o alcance do raio
learning = e a taxa de aprendizado
random_seed = para ter resultados iguais 
"""
tamanho = base.shape

n_neurons = int(math.sqrt(round(5 * math.sqrt(tamanho[0]))))

som = MiniSom(x=n_neurons, y=n_neurons, input_len=4, sigma=1.0, learning_rate=0.5, random_seed=0)

# treino
som.random_weights_init(previsores)
som.train_random(data=previsores, num_iteration=100)

# vizualizando
pcolor(som.distance_map().T)
colorbar()
# plt.show()

"""
daqui podemos ver que os amarelados d mais, proximos a cores mais escuras, com muito contraste pode ser frauds
"""

markers = ['o', 's']
color = ['r', 'g']

"""
bolinha credito aprovado
quadradinho credito não aprovado
"""

# prevendo
for i, x in enumerate(previsores):
    w = som.winner(x)
    plot(w[0]+0.5, w[1]+0.5, markers[classe[i]],
         markerfacecolor='None', markersize=10,
         markeredgecolor = color[classe[i]], markeredgewidth =2)

# plt.show()

# abre a grade para vizualizar a grede, vamos ver o que tem quadrado em bolinha ao mesmo tempo
mapeamento = som.win_map(previsores)

# o axis deixa a linha um abaixo da outra
suspeitos = np.concatenate((mapeamento[(5,9)], mapeamento[(1,1)]), axis=0)
suspeitos = normalizador.inverse_transform(suspeitos)

print("temos: ", len(suspeitos))

classe_resposta = []
indice_base = []
for id_b,i in enumerate(range(len(base))): # percorre toda a base
    for j in range(len(suspeitos)): # percorre a lista de suspeitos
        if base.iloc[i, 0] == int(round(suspeitos[j, 0])): # acha o cara na base ( comparando os seus dados )
            classe_resposta.append(base.iloc[i,4]) # joga na classe de resposta
            indice_base.append(id_b)

classe_resposta = np.asarray(classe_resposta)
suspeitos_final = np.column_stack((suspeitos, classe_resposta))
suspeitos_final = suspeitos_final[suspeitos_final[:, 4].argsort()]

dataframe = pd.DataFrame(suspeitos_final)
dataframe.to_csv('../data/maps_auto_rganizaveis/suspeitos.csv', index=False, header=True)
tratado = pd.read_csv('../data/maps_auto_rganizaveis/suspeitos.csv')
tratado = tratado.rename(columns={'0': 'id', '1': 'renda', '2': 'idade', '3': 'emprestimo', '4': 'inadimplente'})
# melhor tratamento
tratado['id'] =  tratado['id'].astype(int)
tratado['idade'] =  tratado['idade'].astype(int)
tratado['inadimplente'] =  tratado['inadimplente'].astype(int)
tratado.to_csv('../data/maps_auto_rganizaveis/suspeitos_tratado.csv',float_format='%.2f', index=False)