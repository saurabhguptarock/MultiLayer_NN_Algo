import numpy as np
input_size = 3  # no of features
layers = [4, 3]  # no of neuron in 1st and 2nd layer
output_size = 2


def softmax(a):
    return np.exp(a) / np.sum(np.exp(a), axis=1, keepdims=True)


class NeuralNetwork:
    def __init__(self, input_size, layers, output_size):
        np.random.seed(0)
        model = {}

        model['w1'] = np.random.randn(input_size, layers[0])
        model['b1'] = np.zeros((1, layers[0]))

        model['w2'] = np.random.randn(layers[0], layers[1])
        model['b2'] = np.zeros((1, layers[1]))

        model['w3'] = np.random.randn(layers[1], output_size)
        model['b3'] = np.zeros((1, output_size))
        self.model = model

    def forward(self, x):
        w1, w2, w3 = self.model['w1'], self.model['w2'], self.model['w3']
        b1, b2, b3 = self.model['b1'], self.model['b2'], self.model['b3']

        z1 = np.dot(x, w1) + b1
        a1 = np.tanh(z1)

        z2 = np.dot(a1, w2) + b2
        a2 = np.tanh(z2)

        z3 = np.dot(a2, w3) + b3
        y_ = softmax(z3)
        self.activation_outputs = (a1, a2, y_)
        return y_

    def backward(self, x, y, learning_rate=0.001):
        w1, w2, w3 = self.model['w1'], self.model['w2'], self.model['w3']
        b1, b2, b3 = self.model['b1'], self.model['b2'], self.model['b3']
        a1, a2, y_ = self.activation_outputs
        m = x.shape[0]

        delta3 = y_ - y
        dw3 = np.dot(a2.T, delta3)
        db3 = np.sum(delta3, axis=0) / float(m)

        delta2 = (1 - np.square(a2)) * np.dot(delta3, w3.T)
        dw2 = np.dot(a1.T, delta2)
        db2 = np.sum(delta2, axis=0) / float(m)

        delta1 = (1 - np.square(a1)) * np.dot(delta2, w2.T)
        dw1 = np.dot(x.T, delta1)
        db1 = np.sum(delta1, axis=0) / float(m)

        self.model['w1'] -= learning_rate * dw1
        self.model['b1'] -= learning_rate * db1

        self.model['w2'] -= learning_rate * dw2
        self.model['b2'] -= learning_rate * db2

        self.model['w3'] -= learning_rate * dw3
        self.model['b3'] -= learning_rate * db3

    def predict(self, x):
        y_out = self.forward(x)
        return np.argmax(y_out, axis=1)

    def summary(self):
        w1, w2, w3 = self.model['w1'], self.model['w2'], self.model['w3']
        a1, a2, y_ = self.activation_outputs

        print('W1: ', w1.shape)
        print('a1: ', a1.shape)

        print('W2: ', w2.shape)
        print('a2: ', a2.shape)

        print('W3: ', w3.shape)
        print('Y_: ', y_.shape)


def loss(y_oht, p):
    return (-np.mean(y_oht * np.log(p)))


def one_hot(y, depth):
    m = y.shape[0]
    y_oht = np.zeros((m, depth))
    y_oht[np.arange(m), y] = 1
    return y_oht


from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

x, y = make_circles(n_samples=500, shuffle=True, noise=0.05, random_state=1, factor=0.8)
# plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Accent)
# plt.show()

model = NeuralNetwork(input_size=2, layers=[4, 3], output_size=2)


def train(x, y, model, epochs, learning_rate, logs=True):
    training_loss = []
    classes = 2
    y_oht = one_hot(y, classes)
    for ix in range(epochs):
        y_ = model.forward(x)
        l = loss(y_oht, y_)
        training_loss.append(l)
        model.backward(x, y_oht, learning_rate)

        if logs is True:
            print('Epoch %d Loss %.4f ' % (ix, l))
    return training_loss


losses = train(x, y, model, 5000, 0.001, logs=False)
output = model.predict(x)

print(np.sum(output == y) / y.shape[0])


def plot(model, x, y, cmap=plt.cm.jet):
    xmin, xmax = x[:, 0].min() - 1, x[:, 0].max() + 1
    ymin, ymax = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
    z = model(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()


plot(lambda x: model.predict(x), x, y)
