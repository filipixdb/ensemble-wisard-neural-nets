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




'''
#plotar o desempenho medio de acordo com o numero de learners
#ensemble features iguais treino 0.5 sem repeticao

n_learners = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
acc_bagging = np.array([0.7484, 0.7544, 0.7565, 0.7585, 0.7599, 0.7609, 0.7611, 0.7624, 0.7637, 0.7639, 0.7642, 0.7650, 0.7639, 0.7644, 0.7650, 0.7650, 0.7647, 0.7660, 0.7658, 0.7656, 0.7662, 0.7661, 0.7662, 0.7659, 0.7664, 0.7664, 0.7656, 0.7673, 0.7667, 0.7670, 0.7667])
acc_boost = np.array([0.7434, 0.7523, 0.7509, 0.7545, 0.7561, 0.7575, 0.7577, 0.7594, 0.7589, 0.7603, 0.7590, 0.7597, 0.7604, 0.7601, 0.7600, 0.7613, 0.7603, 0.7613, 0.7610, 0.7602, 0.7609, 0.7623, 0.7613, 0.7615, 0.7610, 0.7614, 0.7618, 0.7620, 0.7622, 0.7611, 0.7629])

plt.plot(n_learners, acc_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(n_learners, acc_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 33, 0.7, 0.8])

plt.xlabel(u"Total base learners")
plt.ylabel(u"Accuracy")

plt.title("Ensemble mean accuracy")

plt.legend()
plt.show()

'''


'''
#plotar melhor desempenho de acordo com o numero de learners
#ensemble features iguais treino 0.5 sem repeticao

n_learners = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
melhor_acc_bagging = np.array([0.752, 0.756, 0.758, 0.760, 0.761, 0.762, 0.762, 0.763, 0.765, 0.765, 0.766, 0.767, 0.765, 0.766, 0.766, 0.767, 0.766, 0.768, 0.767, 0.767, 0.768, 0.768, 0.768, 0.768, 0.768, 0.768, 0.767, 0.769, 0.768, 0.768, 0.769])
melhor_acc_boost = np.array([0.751, 0.757, 0.758, 0.760, 0.761, 0.763, 0.763, 0.764, 0.762, 0.764, 0.764, 0.764, 0.765, 0.764, 0.764, 0.765, 0.763, 0.765, 0.764, 0.763, 0.764, 0.765, 0.765, 0.765, 0.764, 0.764, 0.765, 0.765, 0.765, 0.764, 0.766])

plt.plot(n_learners, melhor_acc_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(n_learners, melhor_acc_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 33, 0.7, 0.8])

plt.xlabel(u"Total base learners")
plt.ylabel(u"Accuracy")

plt.title("Ensemble best accuracy")

plt.legend()
plt.show()

'''




'''
#plotar f1 medio de acordo com o numero de learners
#ensemble features iguais treino 0.5 sem repeticao

n_learners = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
f1_bagging = np.array([0.7291, 0.7355, 0.7358, 0.7389, 0.7388, 0.7406, 0.7399, 0.7421, 0.7428, 0.7432, 0.7433, 0.7442, 0.7424, 0.7438, 0.7439, 0.7442, 0.7434, 0.7449, 0.7445, 0.7444, 0.7450, 0.7450, 0.7449, 0.7450, 0.7452, 0.7454, 0.7444, 0.7463, 0.7452, 0.7458, 0.7455])
f1_boost = np.array([0.7289, 0.7401, 0.7386, 0.7429, 0.7438, 0.7457, 0.7456, 0.7478, 0.7466, 0.7484, 0.7475, 0.7480, 0.7488, 0.7487, 0.7484, 0.7499, 0.7485, 0.7498, 0.7494, 0.7487, 0.7495, 0.7508, 0.7498, 0.7503, 0.7495, 0.7501, 0.7504, 0.7504, 0.7508, 0.7499, 0.7515])

plt.plot(n_learners, f1_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(n_learners, f1_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 33, 0.7, 0.8])

plt.xlabel(u"Total base learners")
plt.ylabel(u"Average F1 score")

plt.title("Ensemble mean average F1 score")

plt.legend()
plt.show()

'''


'''
#plotar melhor f1 de acordo com o numero de learners
#ensemble features iguais treino 0.5 sem repeticao

n_learners = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
melhor_f1_bagging = np.array([0.731, 0.738, 0.739, 0.741, 0.743, 0.744, 0.744, 0.745, 0.747, 0.747, 0.747, 0.748, 0.747, 0.748, 0.749, 0.748, 0.748, 0.749, 0.749, 0.749, 0.750, 0.749, 0.750, 0.750, 0.750, 0.750, 0.750, 0.751, 0.750, 0.750, 0.751])
melhor_f1_boost = np.array([0.733, 0.741, 0.742, 0.745, 0.747, 0.747, 0.748, 0.750, 0.749, 0.750, 0.750, 0.750, 0.751, 0.751, 0.750, 0.752, 0.751, 0.752, 0.752, 0.751, 0.751, 0.753, 0.752, 0.752, 0.752, 0.752, 0.752, 0.752, 0.753, 0.752, 0.754])

plt.plot(n_learners, melhor_f1_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(n_learners, melhor_f1_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 33, 0.7, 0.8])

plt.xlabel(u"Total base learners")
plt.ylabel(u"Average F1 score")

plt.title("Ensemble best average F1 score")

plt.legend()
plt.show()

'''







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

