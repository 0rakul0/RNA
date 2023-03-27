import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

class RNA():
    def __init__(self):
        self.X_censu_treinamento = 0
        self.y_censu_treinamento = 0
        self.X_censu_teste = 0
        self.y_censu_teste = 0

    def open(self):
        with open('data/census.pkl', 'rb') as f:
            self.X_censu_treinamento, self.y_censu_treinamento, self.X_censu_teste, self.y_censu_teste = pickle.load(f)

        print(self.X_censu_treinamento.shape)
        print(self.y_censu_treinamento.shape)

        self.treinamento()

    def treinamento(self):
        rn_census = MLPClassifier(max_iter=1000, tol=0.00010, hidden_layer_sizes=(100,100))
        rn_census.fit(self.X_censu_treinamento, self.y_censu_treinamento)

        self.previsor(rn_census)

    def previsor(self, rn_census):
        prev_x = rn_census.predict(self.X_censu_teste)
        prev_y = self.y_censu_teste

        self.acuracia(prev_y, prev_x, rn_census)

    def acuracia(self, prev_y, prev_x, rn_census):
        acerto = accuracy_score(prev_y, prev_x)
        matrix = classification_report(prev_y, prev_x)
        print(acerto)
        print(matrix)
        self.matrix(rn_census)

    def matrix(self, rn_census):
        cm = ConfusionMatrix(rn_census)
        cm.fit(self.X_censu_treinamento, self.y_censu_treinamento)
        sco = cm.score(self.X_censu_teste, self.y_censu_teste)
        cm.draw()
        print(sco)

if __name__ == "__main__":
    rna = RNA()
    rna.open()