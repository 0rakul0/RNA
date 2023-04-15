"""
deep learn de rede neurais recorrente,
a diferença é que essa consome muita memoria para manter os dados já estudados
"""
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd

# chamar os dados
base = pd.read_csv('../data/bolsa/petr4_treinamento.csv')
base_teste = pd.read_csv('../data/bolsa/petr4_teste.csv')

# retirando elementos com valores nulos
base = base.dropna() # tira os nulos nan
base_treino = base.iloc[:, 1:2].values # a coluna + 1 no caso é a coluna open


# normalização
normaliza = MinMaxScaler(feature_range=(0,1)) # deixa entre 0 e no maximo 1
base_treino_normal = normaliza.fit_transform(base_treino)

# aqui vamos pegar os 4 ultimos registros para prever o 5, esse valorpode ser aumentado
previsores = []
preco_real = []

for i in range(90, 1242): # vamos começar do 90 para baixo
    previsores.append(base_treino_normal[i-90:i, 0]) # pega o inicio no caso o indice 0
    preco_real.append(base_treino_normal[i, 0]) # pega o valor da linha 90

# se abrir o array de dados ele deixa em linhas diagonais
previsores, preco_real = np.array(previsores), np.array(preco_real) # deixa no formato para a rede neural entender

# na documentação temos 3 diemsões sendo estas tempo, intervalos e entradas
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1)) # tempo, intervalo, entrada

regressor = Sequential()

# LSTM guarda informações do passado
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(previsores.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50)) # na penultima camada não tem o return_sequences
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1, activation='linear'))

# o rmsprop é muito indocado para esse tipo de rede
regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

regressor.fit(previsores, preco_real, epochs=100, batch_size=32)

preco_real_teste = base_teste.iloc[:, 1:2].values
base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0) # junta por coluna

# recebe a base completa
entradas = base_completa[len(base_completa) - len(base_teste)-90:].values # começa a testar daqui
entradas = entradas.reshape(-1, 1)
entradas = normaliza.transform(entradas)

X_teste = []
for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0])

X_teste = np.array(X_teste)
# deixa no formato para a RNA
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

previsores = regressor.predict(X_teste)
# tras o valor real
previsores = normaliza.inverse_transform(previsores)

print(previsores.mean())
print(preco_real_teste.mean())