# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


'''
#EXEMPLO
# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()
'''




'''
#plotar o desempenho de acordo com tamanho do endereco
#single learner com todas as features

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

'''
# plotar desempenho de acordo com numero de features
# single learner com 13 bits de endereco

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
'''


#TODO: revisar para plotar os acima tbm para o f1


#plotar o desempenho medio de acordo com o numero de learners
#ensemble features iguais treino 0.5 sem repeticao

#plotar melhor desempenho de acordo com o numero de learners
#ensemble features iguais treino 0.5 sem repeticao


#plotar f1 medio de acordo com o numero de learners
#ensemble features iguais treino 0.5 sem repeticao

#plotar melhor f1 de acordo com o numero de learners
#ensemble features iguais treino 0.5 sem repeticao







'''
#plotar o desempenho medio de acordo com o tamanho do treino
#ensemble features iguais 9 learners

tamanho_treino = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
acc_bagging = np.array([0.7435, 0.7519, 0.757, 0.76, 0.7624, 0.7652, 0.7649, 0.7650, 0.7677])
acc_boost = np.array([0.7315, 0.743, 0.7492, 0.7546, 0.7590, 0.7603, 0.7644, 0.7656, 0.7677])

plt.plot(tamanho_treino, acc_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(tamanho_treino, acc_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 1, 0.7, 0.8])

plt.xlabel(u"Training sample")
plt.ylabel(u"Accuracy")

plt.title("Ensemble mean accuracy")

plt.legend()
plt.show()

'''

'''
#plotar melhor desempenho de acordo com o tamanho do treino
#ensemble features iguais 9 learners

tamanho_treino = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
melhor_acc_bagging = np.array([0.746, 0.754, 0.759, 0.761, 0.763, 0.766, 0.766, 0.765, 0.768])
melhor_acc_boost = np.array([0.741, 0.752, 0.756, 0.760, 0.763, 0.764, 0.766, 0.766, 0.768])

plt.plot(tamanho_treino, melhor_acc_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(tamanho_treino, melhor_acc_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 1, 0.7, 0.8])

plt.xlabel(u"Training sample")
plt.ylabel(u"Accuracy")

plt.title("Ensemble best accuracy")

plt.legend()
plt.show()

'''





'''
#plotar f1 medio de acordo com o tamanho do treino
#ensemble features iguais 9 learners

tamanho_treino = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
f1_bagging = np.array([0.7158, 0.7267, 0.7334, 0.7378, 0.7421, 0.7459, 0.7472, 0.7481, 0.7516])
f1_boost = np.array([0.7172, 0.7299, 0.737, 0.7431, 0.7473, 0.7488, 0.7509, 0.7503, 0.7517])

plt.plot(tamanho_treino, f1_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(tamanho_treino, f1_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 1, 0.7, 0.8])

plt.xlabel(u"Training sample")
plt.ylabel(u"Average F1 score")

plt.title("Ensemble mean average F1 score")

plt.legend()
plt.show()

'''


'''
#plotar melhor f1 de acordo com o tamanho do treino
#ensemble features iguais 9 learners

tamanho_treino = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
melhor_f1_bagging = np.array([0.723, 0.731, 0.737, 0.741, 0.745, 0.750, 0.750, 0.751, 0.755])
melhor_f1_boost = np.array([0.723, 0.735, 0.739, 0.745, 0.749, 0.751, 0.753, 0.753, 0.755])

plt.plot(tamanho_treino, melhor_f1_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(tamanho_treino, melhor_f1_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 1, 0.7, 0.8])

plt.xlabel(u"Training sample")
plt.ylabel(u"Average F1 score")

plt.title("Ensemble best average F1 score")

plt.legend()
plt.show()

'''

