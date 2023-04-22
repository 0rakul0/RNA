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

base = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,0,1,1,1], [0,0,1,1,0,1]])

# não passar de 5000 epochs
rbm.train(base, max_epochs=5000)

print(rbm.weights)
"""

[[ 1.08221081e+01 -3.70850576e-01 -3.94196153e-01] valor 0
 [-3.64826958e+00  4.64263899e+00  4.62116649e+00] valor 1
 [-4.02069810e+00  2.32278663e+00  2.38335011e+00] valor 1
 [ 7.33974799e-02  3.10009016e+00  2.95120985e+00] valor 1
 [ 4.64689579e+00 -5.27732152e+00 -5.12425618e+00] valor 0
 [-9.35217917e-03 -3.23751697e+00 -3.22441932e+00] valor 0
 [ 4.64767429e+00 -5.24468304e+00 -5.15796906e+00] valor 0
 ] 
"""