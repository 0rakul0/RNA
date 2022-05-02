from numpy import dot, array, random
from typing import List

import math

from tqdm import tqdm


def step_function(x: float) -> float:
    return 1 if x >= 0 else 0


def percepton_output(weights: array, bias: float, x: array) -> float:
    """
    :param weights: inclui o termo de vies,
    :param bias: inclui o termo de bias
    :param x: inclui o termo de entrada
    :return: retorna o resultado da função de ativação
    """
    calculation = dot(weights, x) + bias
    return step_function(calculation)


def sigmoid(t: float) -> float:
    """
    :param t: valor de entrada
    :return: retorna o valor da função sigmoid
    """
    return 1 / (1 + math.exp(-t))


def neuron_output(weights: array, inputs: array) -> float:
    """
    :param weights: inclui o termo de vies, 
    :param inputs: inclui o termo de bias
    :return: retorna o resultado da função de ativação
    """
    return sigmoid(dot(weights, inputs))


def feed_ford(r_n: List[List[array]], input_array: array) -> List[array]:
    """
    :param r_n: rede_neural
    :param input_array: entradas da rede neural
    :return: valores da rede
    """
    outputs: List[array] = []
    for camada in r_n:
        input_em_bias = input_array + [1]
        output = [neuron_output(neuron, input_em_bias) for neuron in camada]
        outputs.append(output)
        input_array = output

    return outputs


def sqerror_gradient(rede: List[List[array]], x: array, y: array) -> List[array]:
    """
    :param rede: rede neural
    :param x: entrada
    :param y: saida desejada
    :return: retorna o gradiente da rede
    """
    ocultas_grad: List[array] = []
    saidas_grad: List[array] = []
    ocultas_outputs: List[array] = feed_ford(rede, x)
    saidas_outputs: List[array] = feed_ford(rede, x)
    saidas_grad.append([saida_output * (1 - saida_output) * (saida_output - y[i]) for i, saida_output in
                        enumerate(saidas_outputs[-1])])
    for i in range(len(rede) - 2, -1, -1):
        ocultas_grad.append(
            [saida_output * (1 - saida_output) * dot(saidas_grad[-1], rede[i + 1][0]) for saida_output in
             ocultas_outputs[i]])
    return ocultas_grad, saidas_grad


learning_rate = 1


def train(rede: List[List[array]], xs: List[array], ys: List[array], learning_rate: float) -> List[List[array]]:
    """
    :param rede: rede neural
    :param xs: entradas de treinamento
    :param ys: saidas desejadas
    :param learning_rate: taxa de aprendizado
    :return: retorna a rede neural treinada
    """
    for x, y in zip(xs, ys):
        ocultas_grad, saidas_grad = sqerror_gradient(rede, x, y)
        for i, oculta_grad in enumerate(ocultas_grad):
            rede[i][0] = [w - learning_rate * oculta_grad[i] for w in rede[i][0]]
        rede[-1][0] = [w - learning_rate * saidas_grad[0][i] for w in rede[-1][0]]
    return rede


def fizz_buzz_encode(x: int) -> array:
    if x % 15 == 0:
        return [1, 0, 0, 0]
    if x % 5 == 0:
        return [0, 1, 0, 0]
    if x % 3 == 0:
        return [0, 0, 1, 0]
    return [0, 0, 0, 1]


def binary_encode(x: int) -> array:
    binary: List[float] = []
    for i in range(8):
        binary.append(x % 2)
        x = x // 2
    return binary


xs = [binary_encode(n) for n in range(101, 1024)]
ys = [fizz_buzz_encode(n) for n in range(101, 1024)]

NUM_HIDDEN = 25

rede = [[[random.random() for _ in range(10 + 1)] for _ in range(NUM_HIDDEN)],
        [[random.random() for _ in range(NUM_HIDDEN + 1)] for _ in range(4)]
        ]


def argmax(array: array) -> int:
    return max(array.index(max(array)), key=lambda x: x[1])


num_correct = 0

for n in range(1, 101):
    x = binary_encode(n)
    predicted = argmax(feed_ford(rede, x)[-1])
    actual = argmax(fizz_buzz_encode(n))
    labels = [str(n), 'fizz', 'buzz', 'fizzbuzz']
    print(f'{n}, {labels[predicted]} as {labels[actual]}')

    if predicted == actual:
        num_correct += 1
print(num_correct, "/", 100)
