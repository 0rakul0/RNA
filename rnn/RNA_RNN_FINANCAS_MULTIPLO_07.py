"""
deep learn de rede neurais recorrente,
com multiplas saidas
o objetivo é ver a maxima em um dia
"""
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# chamar os dados
base = pd.read_csv('../data/bolsa/petr4_treinamento.csv')

# retirando elementos com valores nulos
base = base.dropna() # tira os nulos nan
base_treino = base.iloc[:, 1:2].values # a coluna + 1 tirando a ultima coluna

base_valor_maximo = base.iloc[:, 2:3].values

# normalização
normaliza = MinMaxScaler(feature_range=(0,1)) # deixa entre 0 e no maximo 1
base_treino_normal = normaliza.fit_transform(base_treino)
base_valor_maximo_normal = normaliza.fit_transform(base_valor_maximo)


# aqui vamos pegar os 4 ultimos registros para prever o 5, esse valorpode ser aumentado
previsores = []
preco_real_base = []
preco_real_maximo = []

for i in range(90, 1242): # vamos começar do 90 para baixo
    previsores.append(base_treino_normal[i-90:i, 0]) # pega o inicio no caso o indice 0
    preco_real_base.append(base_treino_normal[i, 0]) # pega o valor da linha 90
    preco_real_maximo.append(base_valor_maximo_normal[i, 0])

# se abrir o array de dados ele deixa em linhas diagonais
previsores, preco_real, preco_real_maximo = np.array(previsores), np.array(preco_real_base), np.array(preco_real_maximo ) # deixa no formato para a rede neural entender

# quando temos só um atributo previso precisamos deixa-lo com 3D, sendo essas: tempo, intervalos e entradas
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1)) # tempo, intervalo, entrada

# precisamos juntar os valores de real e real_maximo
preco_real = np.column_stack((preco_real_base, preco_real_maximo))

regressor = Sequential()

# LSTM guarda informações do passado
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(previsores.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50)) # na penultima camada não tem o return_sequences
regressor.add(Dropout(0.2))
regressor.add(Dense(units=2, activation='linear'))

# o rmsprop é muito indocado para esse tipo de rede
regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

# usamos isso para não perder tempo na modelagem de uma rede neural assim ele já dá as epocas minimas de treino
es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
# esse cara grava o melhor peso
mcp = ModelCheckpoint(filepath='peso.h5', monitor='loss', save_best_only=True, verbose=1)

regressor.fit(previsores, preco_real, epochs=100, batch_size=32, callbacks=[es, rlr, mcp])

base_teste = pd.read_csv('../data/bolsa/petr4_teste.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values # aqui tem somente os valores de abertura
preco_real_alto = base_teste.iloc[:, 2:3].values # aqui tem somente os de mais alta no dia

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

# usa para testes
X_teste = []
for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0:6])
# joga no formato de array
X_teste = np.array(X_teste)

# usa a rede para fazer a previsão
previsores = regressor.predict(X_teste)
previsores = normaliza.inverse_transform(previsores)


print(f'diferença de {previsores.mean()-preco_real_teste.mean()}')

# gerando um grafico para a rede neural
plt.plot(preco_real_teste, color='red', label='Preço abertura real')
plt.plot(preco_real_alto, color='black', label='Preco em alta')

plt.plot(previsores[:, 0], color='blue', label='previsao do preço previsto')
plt.plot(previsores[:, 1], color='orange', label='Previsão de alta prevista')

plt.title('previsão de preço de ações ')
plt.xlabel('Tempo')
plt.ylabel('Valor yahoo')
plt.legend()
plt.show()