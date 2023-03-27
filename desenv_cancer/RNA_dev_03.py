"""
adicionando mais camadas ocultas
"""
import pandas as pd
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# importa os dados já tratados
previsores = pd.read_csv('../data/breast_cancer/entradas_breast.csv')
classe = pd.read_csv('../data/breast_cancer/saidas_breast.csv')

"""
vale lembrar que previssores estão as colunas já codificadas e classe é as respostas
a função train_teste_split serve para separar para teste e treinamento
no caso aqui vai ser uma classificação binaria, sim ou não
"""

# separa a amostra em treinamento e teste, tanto de previsores quanto teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

"""
o tipo de Sequencial significa que teremos uma sequencia de camadas entrada -> ocultas -> saida
o Dense é quando todos eles estão altamente conectados passando de camada para camada a frente
"""

# chama o sequenciador
classificador = Sequential()

"""
claculo para saber quantos neuronios usar para a camanda oculta basta somar entrada e saida e dividir por dois
entrada é o numero de colunas do dataset, no caso atual é 30
e para saida é 1

logo usamos 30+1 / 2 = 15.5 arredonda para cima = 16

"""
# primeira camada oculta, já deixando conectada
classificador.add(Dense(units=16, # quantidade de neuronios nas
                        activation='relu', # ativador
                        kernel_initializer='random_uniform', # aleatoriedade
                        input_dim=30 # quantidade de colunas de entrada
                        ))

# para add mais camadas ocultas basta repetir mas sem o input_dim nem sempre fica melhor.. pode chegar a piorar
classificador.add(Dense(units=16, # quantidade de neuronios nas
                        activation='relu', # ativador
                        kernel_initializer='random_uniform', # aleatoriedade
                         ))

# camada de saida
classificador.add(Dense(units=1,# proximo de zero é cancer benigno, proximo de 1 é cancer ruim
                        activation='sigmoid'
                        ))

"""
para compilar o modelo, basta chamar o compile e para treinar usar o fit passando a base e juste dos pesos
"""

otimizador = keras.optimizers.Adam(learning_rate=0.001, decay=0.001, clipvalue=0.5)

# compilador
classificador.compile(optimizer=otimizador,
                      loss='binary_crossentropy', # usado para classificadores binarios
                      metrics= ['binary_accuracy']) # como a resposta é binaria é recomendado usar esse)

# treinamento
classificador.fit(previsores_treinamento, classe_treinamento, # passa a parte de treinamento
                  batch_size = 10,# faz o ajuste dos pesos de 10 em 10, as vezes é bom diminuir para ter melhor resultado
                  epochs= 100)

"""
testando e vizualizando os pesos
"""

peso0 = classificador.layers[0].get_weights()
peso1 = classificador.layers[1].get_weights()
peso2 = classificador.layers[2].get_weights()

print(len(peso0))
print(len(peso1))
print(len(peso2))

"""
para testar se estar eficiente, usamos o predict para testar como é zero ou um, usamos o previsões > que 0.5
caso maior é true, caso menor retorna falso
para vizualizar usamos a matriz de confusão
"""

# teste
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

# aqui passamos o que foi testado e vemos se está bom ou não
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

resultado = classificador.evaluate(previsores_teste, classe_teste)

print('usando o skitlearn: ',precisao)
print('usando o skitlearn: \n',matriz)

print('usando o keras: ',resultado)


