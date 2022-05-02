import pickle
from sklearn.neural_network import MLPClassifier

with open('data/credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

#print(X_credit_treinamento.shape)
#print(y_credit_treinamento.shape)

rede_neral_credit = MLPClassifier(max_iter=1000, verbose=True, hidden_layer_sizes=(10, 10, 10), random_state=1)
rede_neral_credit.fit(X_credit_treinamento, y_credit_treinamento)
