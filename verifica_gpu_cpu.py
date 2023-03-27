import tensorflow as tf

# cria um tensor constante e o imprime
hello = tf.constant('Olá, TensorFlow!')
print(hello)

# verifica a versão do TensorFlow
print('Versão do TensorFlow:', tf.__version__)

# verifica se a GPU está disponível
if tf.test.is_gpu_available():
    print('GPU disponível')
else:
    print('GPU indisponível')
