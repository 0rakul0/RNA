from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot
import matplotlib.pyplot as plt

base = pd.read_csv('../data/maps_auto_rganizaveis/wines.csv')

print(base.shape)

previsores = base.iloc[:, 1:14].values
classe = base.iloc[:,0].values

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
som = MiniSom(x=8, y=8, input_len=13, sigma=1.0, learning_rate=0.5, random_seed=1)

# treino
som.random_weights_init(previsores)
som.train_random(data=previsores, num_iteration=100)

# vizualizando
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's', 'd']
color = ['r', 'g', 'b']

# transforma a classe
classe[classe==1]=0
classe[classe==2]=1
classe[classe==3]=2

# prevendo
for i, x in enumerate(previsores):
    w = som.winner(x)
    plot(w[0]+0.5, w[1]+0.5, markers[classe[i]],
         markerfacecolor='None', markersize=10,
         markeredgecolor = color[classe[i]], markeredgewidth =2)

plt.show()