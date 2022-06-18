import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

class RNA():
    def __init__(self):
        self.X_credit_treinamento = 0
        self.y_credit_treinamento = 0
        self.X_credit_teste = 0
        self.y_credit_teste = 0

    def open(self):
        with open('data/credit.pkl', 'rb') as f:
            self.X_credit_treinamento, self.y_credit_treinamento, self.X_credit_teste, self.y_credit_teste = pickle.load(f)

        print(self.X_credit_treinamento.shape)
        print(self.y_credit_treinamento.shape)
        self.treinamento()

    def treinamento(self):
        # 10, 10, 10 <- renda, idade e divida :::: para vizualizar verbose=True,
        rede_neral_credit = MLPClassifier(max_iter=1000, hidden_layer_sizes=(30,30, 30),
                                          random_state=1, solver='adam', activation='relu')
        rede_neral_credit.fit(self.X_credit_treinamento, self.y_credit_treinamento)
        self.previsoes(rede_neral_credit)

    def previsoes(self, rna):
        prev = rna.predict(self.X_credit_teste)
        self.metricas(prev, rna)

    def metricas(self, prev, rna):
        accuracy_score(self.y_credit_teste, prev)
        matrix = classification_report(self.y_credit_teste, prev)
        print(matrix)
        self.matrix(rna)

    def matrix(self, rna):
        cm = ConfusionMatrix(rna)
        cm.fit(self.X_credit_treinamento, self.y_credit_treinamento)
        cm.score(self.X_credit_teste, self.y_credit_teste)


if __name__ == "__main__":
    rna = RNA()
    rna.open()

