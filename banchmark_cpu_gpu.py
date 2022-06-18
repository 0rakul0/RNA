import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from time import time


def conv_model(device, filters, kz):
    with tf.device(device):
        model = Sequential()
        model.add(Conv2D(filters, kernel_size=kz, padding='same', activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPool2D(pool_size=2))
        model.add(Conv2D(filters, kernel_size=kz, padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=2))
        model.add(Conv2D(filters, kernel_size=kz, padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        model.summary()
        return model

def run(model, x_train, y_train, epochs=128, batch_size=32):
    start = time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    end = time()
    return end - start


(x_train, y_train), (_, _) = mnist.load_data()
x_train = x_train/255
x_train = x_train.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, num_classes = 10)

cpu_model = conv_model('CPU', 64, 3)
gpu_model = conv_model('DML', 64, 3)

epochs = 8
bz = 64

conv_cpu_time = run(cpu_model, x_train=x_train, y_train=y_train, epochs=epochs, batch_size=bz)
conv_gpu_time = run(gpu_model, x_train=x_train, y_train=y_train, epochs=epochs, batch_size=bz)

print('tempo de cpu {} epochs: {}'.format(epochs, conv_cpu_time))
print('tempo de gpu {} epochs: {}'.format(epochs, conv_gpu_time))