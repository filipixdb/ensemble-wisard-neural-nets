# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


'''
# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()
'''


'''

#plotar o desempenho de acordo com tamanho do endereco
tam_endereco = list(xrange(31))
tam_endereco = tam_endereco[1:]

acuracia = [0.306, 0.336, 0.381, 0.430, 0.507, 0.580, 0.636, 0.681, 0.704, 0.725, 0.731, 0.740, 0.742, 0.742, 0.741, 0.739, 0.742, 0.729, 0.730, 0.730, 0.725, 0.712, 0.699, 0.686, 0.683, 0.647, 0.645, 0.622, 0.625, 0.587]

plt.plot(tam_endereco, acuracia, 'ro', linestyle='-', label="WiSARD")
plt.axis([0, 31, 0, 1])

plt.xlabel(u"Address size")
plt.ylabel(u"Accuracy")

plt.title("General accuracy")

plt.legend()
plt.show()

'''


# plotar desempenho de acordo com numero de features

numero_features = list(xrange(21))
numero_features = numero_features[1:]

acuracia = [ 0.3, 0.307, 0.421, 0.534, 0.583, 0.630, 0.684, 0.711, 0.731, 0.737, 0.740, 0.743, 0.746, 0.748, 0.75, 0.748, 0.747, 0.746, 0.745, 0.740]

plt.plot(numero_features, acuracia, 'ro', linestyle='-', label="WiSARD")
plt.axis([0, 21, 0, 1])

plt.xlabel(u"Number of features")
plt.ylabel(u"Accuracy")

plt.title("General accuracy")

plt.legend()
plt.show()
