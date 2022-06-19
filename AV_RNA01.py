import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold

import numpy as np
import pickle

class AV_RNA():
    def __init__(self):
        self.X_credit_treinamento = 0
        self.y_credit_treinamento = 0
        self.X_credit_teste = 0
        self.y_credit_teste = 0
        self.resultados_arvore = []
        self.resultados_random_arvore = []
        self.resultados_knn = []
        self.resultados_regressao_logistica = []
        self.resultados_svc = []
        self.resultados_rna = []

    def open(self):
        with open('data/credit.pkl', 'rb') as f:
            self.X_credit_treinamento, self.y_credit_treinamento, self.X_credit_teste, self.y_credit_teste = pickle.load(f)

        print(self.X_credit_treinamento.shape, self.y_credit_treinamento.shape)
        print(self.X_credit_teste.shape, self.y_credit_teste.shape)

    def concatnate(self):
        X_credit = np.concatenate((self.X_credit_treinamento, self.X_credit_teste), axis=0)
        y_credit = np.concatenate((self.y_credit_treinamento, self.y_credit_teste), axis=0)
        # self.arvoreDesicao(X_credit,y_credit)
        # self.randomFlorest(X_credit, y_credit)
        # self.knn_classicator(X_credit,y_credit)
        # self.regressaoLogistica(X_credit,y_credit)
        # self.svc(X_credit, y_credit)
        # self.rna(X_credit, y_credit)
        self.validador_arvore_desicao(X_credit, y_credit)
        self.validador_random_florest(X_credit, y_credit)
        self.validador_knn(X_credit, y_credit)
        self.validador_regressao_logistica(X_credit, y_credit)
        self.validador_svc(X_credit, y_credit)
        self.validador_rna(X_credit, y_credit)
        self.df_cross_valid()

    def arvoreDesicao(self, X, y):
        parametros = {'criterion': ['gini', 'entropy'],
                     'splitter':['best', 'random'],
                     'min_samples_split': [2,5,10],
                     'min_samples_leaf': [1,5,10],
                     }
        grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parametros)
        grid_search.fit(X, y)

        melhores_prams = grid_search.best_params_
        melhor_result = grid_search.best_score_

        # print("\n##### arvore de desição #####")
        # print(f"melhores parametros: {melhores_prams}")
        # print(f"melhores resultados: {melhor_result}")

    def randomFlorest(self, X, y):
        parametros = {'criterion': ['gini', 'entropy'],
                      'n_estimators':[10,40,100,150 ],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 5, 10],
                      }

        grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parametros)

        grid_search.fit(X, y)

        melhores_prams = grid_search.best_params_
        melhor_result = grid_search.best_score_

        # print("\n##### random florest #####")
        # print(f"melhores parametros: {melhores_prams}")
        # print(f"melhores resultados: {melhor_result}")

    def knn_classicator(self, X,y):
        parametros = {
            'n_neighbors':[3,5,10,20],
            'p':[1,2]
        }
        grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parametros)

        grid_search.fit(X, y)

        melhores_prams = grid_search.best_params_
        melhor_result = grid_search.best_score_

        # print("\n##### knn #####")
        # print(f"melhores parametros: {melhores_prams}")
        # print(f"melhores resultados: {melhor_result}")

    def regressaoLogistica(self, X,y):
        parametros = {
            'tol':[0.0001, 0.00001,0.000001],
            'C': [1.0,1.5,2.0],
            'solver': ['lbfgs', 'sag', 'saga']
        }
        grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=parametros)

        grid_search.fit(X, y)

        melhores_prams = grid_search.best_params_
        melhor_result = grid_search.best_score_

        # print("\n##### regressão logistica #####")
        # print(f"melhores parametros: {melhores_prams}")
        # print(f"melhores resultados: {melhor_result}")

    def svc(self, X,y):
        parametros = {
            'tol': [0.001, 0.0001, 0.00001],
            'C': [1.0,1.5,2.0],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
        }
        grid_search = GridSearchCV(estimator=SVC(), param_grid=parametros)

        grid_search.fit(X, y)

        melhores_prams = grid_search.best_params_
        melhor_result = grid_search.best_score_

        # print("\n##### SVC #####")
        # print(f"melhores parametros: {melhores_prams}")
        # print(f"melhores resultados: {melhor_result}")

    def rna(self, X,y):
        parametros = {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'batch_size':[10,56]
        }
        grid_search = GridSearchCV(estimator=MLPClassifier(max_iter=500), param_grid=parametros, cv=3,
                           scoring='accuracy')

        grid_search.fit(X, y)
        melhores_prams = grid_search.best_params_
        melhor_result = grid_search.best_score_

        # print("\n##### SVC #####")
        # print(f"melhores parametros: {melhores_prams}")
        # print(f"melhores resultados: {melhor_result}")

    def validador_arvore_desicao(self,X,y):
        for i in range(30):
            kf = KFold(n_splits=10, shuffle=True, random_state=i)
            arvore = DecisionTreeClassifier(criterion='entropy', min_samples_leaf= 1,
                                            min_samples_split=2, splitter='random')

            score = cross_val_score(arvore, X, y, cv=kf)
            # print(f"validacao cruzada: {score}")
            self.resultados_arvore.append(score.mean())

    def validador_random_florest(self,X,y):
        for i in range(30):
            kf = KFold(n_splits=10, shuffle=True, random_state=i)
            arvore = RandomForestClassifier(criterion='entropy', min_samples_leaf= 1,
                                            min_samples_split=2, n_estimators=100)

            score = cross_val_score(arvore, X, y, cv=kf)
            # print(f"validacao cruzada: {score}")
            self.resultados_random_arvore.append(score.mean())

    def validador_knn(self, X,y):
        for i in range(30):
            kf = KFold(n_splits=10, shuffle=True, random_state=i)
            vali_knn = KNeighborsClassifier(n_neighbors= 20, p= 1)

            score = cross_val_score(vali_knn, X, y, cv=kf)
            # print(f"validacao cruzada: {score}")
            self.resultados_knn.append(score.mean())

    def validador_regressao_logistica(self, X,y):
        for i in range(30):
            kf = KFold(n_splits=10, shuffle=True, random_state=i)
            vali_regressao_logistica = LogisticRegression(C= 1.0, solver= 'lbfgs', tol= 0.0001)

            score = cross_val_score(vali_regressao_logistica, X, y, cv=kf)
            # print(f"validacao cruzada: {score}")
            self.resultados_regressao_logistica.append(score.mean())

    def validador_svc(self, X,y):
        for i in range(30):
            kf = KFold(n_splits=10, shuffle=True, random_state=i)
            vali_svc = SVC(C = 1.5, kernel ='rbf', tol = 0.001)

            score = cross_val_score(vali_svc, X, y, cv=kf)
            # print(f"validacao cruzada: {score}")
            self.resultados_svc.append(score.mean())

    def validador_rna(self, X, y):
        for i in range(30):
            kf = KFold(n_splits=10, shuffle=True, random_state=i)
            vali_rna = MLPClassifier(activation='relu', batch_size= 10, solver= 'adam')
            score = cross_val_score(vali_rna, X, y, cv=kf)
            print(f"interações: {i}")
            self.resultados_rna.append(score.mean())

    def df_cross_valid(self):
        resultados = pd.DataFrame({'Arvore':self.resultados_arvore, 'RandomFlorest':self.resultados_random_arvore,
                                   'knn':self.resultados_knn, 'RegressãoLogistica':self.resultados_regressao_logistica,
                                   'svc':self.resultados_svc, 'rna':self.resultados_rna})
        print(resultados)
        resultados.to_csv("./data/resultados.csv")
        with open('./data/resultado.pkl', mode='wb') as f:
            pickle.dump([self.resultados_arvore, self.resultados_random_arvore, self.resultados_knn,
                         self.resultados_regressao_logistica, self.resultados_svc, self.resultados_rna], f)


if __name__ == "__main__":
    rna = AV_RNA()
    rna.open()
    rna.concatnate()

