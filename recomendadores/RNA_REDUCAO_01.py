import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, metrics

# esse aqui é uma rede neural circular
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, Parallel


# importando o dataset
base = datasets.load_digits()

previsores = np.asarray(base.data, 'float32')
classe = base.target

# normalização
normaliza = MinMaxScaler(feature_range=(0,1))
previsores = normaliza.fit_transform(previsores)

# divisão da base de dados
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe,
                                                                                              test_size=0.2,
                                                                                              random_state=0)

rbm = BernoulliRBM(random_state=0)
# numero de interações
rbm.n_iter = 25

# numero de neurons na camada oculta
rbm.n_components = 50

# naive_baies
naive_rbm = GaussianNB()

# executa uma linha logica primeiro o rbm e com a resposta o naive
classificador_rbm = Pipeline(steps=[('rbm', rbm), ('naive', naive_rbm)])

classificador_rbm.fit(previsores_treinamento, classe_treinamento)

# vizualição
plt.figure(figsize=(20,20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10,10, i+1)
    plt.imshow(comp.reshape((8,8)), cmap=plt.cm.gray_r)
    plt.xticks()
    plt.yticks()

plt.show()

previsoes_rbm = classificador_rbm.predict(previsores_teste)
previsoes_rbm = metrics.accuracy_score(previsoes_rbm, classe_teste)
print('pelo modelo de rbm',previsoes_rbm)

naive = GaussianNB()
naive.fit(previsores_treinamento, classe_treinamento)
previsoes_naive = naive.predict(previsores_teste)
previsoes_naive = metrics.accuracy_score(previsoes_naive, classe_teste)
print('pelo naive_bayes', previsoes_naive)