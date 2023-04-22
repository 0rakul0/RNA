from rbm import RBM
import numpy as np

"""
o modelo é autoalimentado por ele mesmo, no caso as entradas também é uma saida
no caso funciona por pontos, quem tem mais pontos maior a chance de aparecer mais
"""
rbm = RBM(num_visible=6, num_hidden=2)

"""
filmes de terror - titulos

terror, terror, terror, comedia, comedia, comedia
"""

base = np.array([[1,1,1,0,0,0],
                 [1,0,1,0,0,0],
                 [1,1,1,0,0,0],
                 [0,0,1,1,1,1],
                 [0,0,1,1,0,1],
                 [0,1,1,1,0,1],
                 [0,1,1,1,1,1],
                 [1,1,1,1,1,1],
                 ])

filmes = ['A bruxa', 'invocação do mal', 'o chamado',
          'mario', 'gente grande', 'amarican pie']

# não passar de 5000 epochs
rbm.train(base, max_epochs=5000)

print(rbm.weights)
"""
[[ 9.88934323 -0.41252554 -0.41844389]
 [-3.65963068  4.58692161  4.73872338]
 [-3.65974645  4.59471148  4.7314382 ]
 [ 0.07246456  3.0976708   3.02420027]
 [ 5.72427563 -3.2752295  -3.13411383]
 [ 4.6036344  -5.10684747 -5.20776848]
 [ 4.60586407 -5.1294529  -5.18974999]]
"""
usuario_1 = np.array([[1,1,0,0,0,0]])
usuario_2 = np.array([[0,0,0,0,1,1]])

resposta1 = rbm.run_visible(usuario_1)
resposta2 = rbm.run_visible(usuario_2)

camada_oculta = np.array(resposta2)
recomendacao = rbm.run_hidden(camada_oculta)

for i in range(len(usuario_2[0])):
    if usuario_2[0,i] == 0 and recomendacao[0,i] ==1:
        print(filmes[i])
