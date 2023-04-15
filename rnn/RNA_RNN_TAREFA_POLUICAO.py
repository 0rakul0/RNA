"""
deep learn de rede neurais recorrente,
No exemplo sobre a previsão dos valores das ações, nós trabalhamos com uma série temporal que apresenta os valores dia a dia.
Porém, também podemos utilizar outros formatos de data, como por exemplo: horas, minutos ou segundos (dependendo do contexto).
A base de dados desta tarefa possui essas características,
 na qual temos em cada registro o ano, o mês, o dia e a hora juntamente com o valor de poluição naquele momento e algumas características climáticas.
  Na imagem abaixo você pode visualizar alguns registros


O atributo No  é somente a contagem de registros (como uma chave primária), o year , month , day  e hour
indicam a dimensão temporal (de hora em hora); o atributo pm2.5  diz respeito ao nível de poluição (que faremos a previsão) e por fim,
todos os outros atributos serão os previsores. Baseado nos atributos previsores, a ideia é indicar o nível de poluição em uma determinada hora.

Siga as seguintes dicas para essa atividade:

Use a função dropna()  para excluir valores faltantes
Os atributos No , year , month , day , hour  e cbwd  devem ser excluídos,
pois em uma série temporal essas informações não são importantes (o cbwd  é somente um campo string)
Este é um problema com uma única saída (pm2.5 ) e múltiplos previsores
Você pode testar com vários valores de intervalos de tempo (começando com 10, por exemplo)
Depois de treinar a rede neural, gere o gráfico para visualizar os resultados das previsões

Bom trabalho!
"""
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# chamar os dados
base = pd.read_csv('../data/tarefas/poluicao.csv')

# retirando elementos com valores nulos
base = base.dropna() # tira os nulos nan

# tira as tabelas que não irão ser usadas
base = base.drop('No', axis = 1)
base = base.drop('year', axis = 1)
base = base.drop('month', axis = 1)
base = base.drop('day', axis = 1)
base = base.drop('hour', axis = 1)
base = base.drop('cbwd', axis = 1)

# Os atributos previsores são todos menos o índice 0
base_treinamento = base.iloc[:, 1:7].values

# Busca dos valores que será feita a previsão, ou seja o primeiro atributo pm2.5
poluicao = base.iloc[:, 0].values

# normalização
normaliza = MinMaxScaler(feature_range=(0,1)) # deixa entre 0 e no maximo 1
base_treinamento_normalizada = normaliza.fit_transform(base_treinamento)

# Necessário mudar o formato da variável para pode aplicar a normalização
poluicao = poluicao.reshape(-1, 1)
poluicao_normalizado = normaliza.fit_transform(poluicao)

previsores = []
poluicao_real = []

for i in range(10, 41757):
    previsores.append(base_treinamento_normalizada[i-10:i, 0:6])
    poluicao_real.append(poluicao_normalizado[i, 0])
previsores, poluicao_real = np.array(previsores), np.array(poluicao_real)

# inicio da rede
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
regressor.add(Dense(units=2, activation='linear'))

# o rmsprop é muito indocado para esse tipo de rede
regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

# usamos isso para não perder tempo na modelagem de uma rede neural assim ele já dá as epocas minimas de treino
es = EarlyStopping(monitor='loss', min_delta=0.0026, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
# esse cara grava o melhor peso
mcp = ModelCheckpoint(filepath='peso.h5', monitor='loss', save_best_only=True, verbose=1)

regressor.fit(previsores, poluicao_real, epochs=100, batch_size=32, callbacks=[es, rlr, mcp])

previsoes = regressor.predict(previsores)
previsoes = normaliza.inverse_transform(previsoes)

# Geração do gráfico. Será gerado um gráfico de barras porque temos muitos registros
plt.plot(poluicao, color = 'red', label = 'Poluição real')
plt.plot(previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão poluição')
plt.xlabel('Horas')
plt.ylabel('Valor poluição')
plt.legend()
plt.show()
