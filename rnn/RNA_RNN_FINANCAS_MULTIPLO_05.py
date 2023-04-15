"""
deep learn de rede neurais recorrente,
com multiplas saidas
"""
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# chamar os dados
base = pd.read_csv('../data/bolsa/petr4_treinamento.csv')

# retirando elementos com valores nulos
base = base.dropna() # tira os nulos nan
base_treino = base.iloc[:, 1:7].values # a coluna + 1 tirando a ultima coluna

# normalização
normaliza = MinMaxScaler(feature_range=(0,1)) # deixa entre 0 e no maximo 1
base_treino_normal = normaliza.fit_transform(base_treino)

# aqui vamos pegar os 4 ultimos registros para prever o 5, esse valorpode ser aumentado
previsores = []
preco_real = []

for i in range(90, 1242): # vamos começar do 90 para baixo
    previsores.append(base_treino_normal[i-90:i, 0:6]) # pega o inicio no caso o indice 0
    preco_real.append(base_treino_normal[i, 0]) # pega o valor da linha 90

# se abrir o array de dados ele deixa em linhas diagonais
previsores, preco_real = np.array(previsores), np.array(preco_real) # deixa no formato para a rede neural entender

regressor = Sequential()

# LSTM guarda informações do passado
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(previsores.shape[1], 6)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50)) # na penultima camada não tem o return_sequences
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1, activation='sigmoid'))

# o rmsprop é muito indocado para esse tipo de rede
regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

# usamos isso para não perder tempo na modelagem de uma rede neural assim ele já dá as epocas minimas de treino
es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
# esse cara grava o melhor peso
mcp = ModelCheckpoint(filepath='peso.h5', monitor='loss', save_best_only=True, verbose=1)

regressor.fit(previsores, preco_real, epochs=100, batch_size=32, callbacks=[es, rlr, mcp])


