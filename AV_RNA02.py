import pandas as pd
import numpy as np
import pickle

class AV_RNA02():
    def __init__(self):

        self.resultados_arvore = []
        self.resultados_random_arvore = []
        self.resultados_knn = []
        self.resultados_regressao_logistica = []
        self.resultados_svc = []
        self.resultados_rna = []

    def open(self):
        with open('./data/resultado.pkl', 'rb') as f:
            self.resultados_arvore, self.resultados_random_arvore, self.resultados_knn, self.resultados_regressao_logistica, self.resultados_svc, self.resultados_rna = pickle.load(f)

    def ler_csv(self):
        df_resultados = pd.read_csv('./data/resultados.csv', index_col=0)

        print(df_resultados.describe())

        print(df_resultados.var())
        print("### desvio padrão ###")
        print((df_resultados.std()/df_resultados.mean())*100)

if __name__ == "__main__":
    av = AV_RNA02()
    av.ler_csv()