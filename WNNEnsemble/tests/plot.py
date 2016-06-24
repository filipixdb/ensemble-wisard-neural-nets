# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


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

plt.plot(tam_endereco, acuracia, 'ro', linestyle='-', label="Single WiSARD")
plt.axis([0, 31, 0.1, 0.8])

plt.xlabel(u"Address size")
plt.ylabel(u"Accuracy")

plt.title("General accuracy")

plt.legend(loc='lower right')
plt.show()

'''



'''
#plotar o f1 de acordo com tamanho do endereco
#single learner com todas as features

tam_endereco = list(xrange(31))
tam_endereco = tam_endereco[1:]

f1 = [0.153, 0.232, 0.330, 0.418, 0.520, 0.597, 0.647, 0.681, 0.697, 0.711, 0.715, 0.722, 0.723, 0.723, 0.724, 0.724, 0.729, 0.717, 0.721, 0.723, 0.719, 0.711, 0.702, 0.693, 0.690, 0.660, 0.658, 0.638, 0.640, 0.604]

plt.plot(tam_endereco, f1, 'ro', linestyle='-', label="Single WiSARD")
plt.axis([0, 31, 0.1, 0.8])

plt.xlabel(u"Address size")
plt.ylabel(u"Average F1 score")

plt.title("Average F1 score")

plt.legend(loc='lower right')
plt.show()

'''






'''
# plotar desempenho de acordo com numero de features
# single learner com 13 bits de endereco

numero_features = list(xrange(21))
numero_features = numero_features[1:]

acuracia = [ 0.3, 0.307, 0.421, 0.534, 0.583, 0.630, 0.684, 0.711, 0.731, 0.737, 0.740, 0.743, 0.746, 0.748, 0.75, 0.748, 0.747, 0.746, 0.745, 0.740]

plt.plot(numero_features, acuracia, 'ro', linestyle='-', label='Single WiSARD')
plt.axis([0, 21, 0.1, 0.8])

plt.xlabel(u"Number of features")
plt.ylabel(u"Accuracy")

plt.title("General accuracy")

plt.legend(loc='lower right')
plt.show()

'''


'''
# plotar f1 de acordo com numero de features
# single learner com 13 bits de endereco

numero_features = list(xrange(21))
numero_features = numero_features[1:]

f1 = [0.138, 0.158, 0.382, 0.546, 0.601, 0.642, 0.687, 0.708, 0.727, 0.731, 0.733, 0.735, 0.736, 0.738, 0.740, 0.735, 0.733, 0.730, 0.728, 0.721]

plt.plot(numero_features, f1, 'ro', linestyle='-', label='Single WiSARD')
plt.axis([0, 21, 0.1, 0.8])

plt.xlabel(u"Number of features")
plt.ylabel(u"Average F1 score")

plt.title("Average F1 score")

plt.legend(loc='lower right')
plt.show()

'''





'''
#plotar o desempenho medio de acordo com o numero de learners
#ensemble features iguais treino 0.5 sem repeticao

n_learners = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
acc_bagging = np.array([0.7484, 0.7544, 0.7565, 0.7585, 0.7599, 0.7609, 0.7611, 0.7624, 0.7637, 0.7639, 0.7642, 0.7650, 0.7639, 0.7644, 0.7650, 0.7650, 0.7647, 0.7660, 0.7658, 0.7656, 0.7662, 0.7661, 0.7662, 0.7659, 0.7664, 0.7664, 0.7656, 0.7673, 0.7667, 0.7670, 0.7667])
acc_boost = np.array([0.7434, 0.7523, 0.7509, 0.7545, 0.7561, 0.7575, 0.7577, 0.7594, 0.7589, 0.7603, 0.7590, 0.7597, 0.7604, 0.7601, 0.7600, 0.7613, 0.7603, 0.7613, 0.7610, 0.7602, 0.7609, 0.7623, 0.7613, 0.7615, 0.7610, 0.7614, 0.7618, 0.7620, 0.7622, 0.7611, 0.7629])

plt.plot(n_learners, acc_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(n_learners, acc_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 33, 0.7, 0.78])

plt.xlabel(u"Total base learners")
plt.ylabel(u"Accuracy")

plt.title("Ensemble mean accuracy")

plt.legend(loc='lower right')
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
plt.axis([0, 33, 0.70, 0.78])

plt.xlabel(u"Total base learners")
plt.ylabel(u"Accuracy")

plt.title("Ensemble best accuracy")

plt.legend(loc='lower right')
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
plt.axis([0, 33, 0.7, 0.78])

plt.xlabel(u"Total base learners")
plt.ylabel(u"Average F1 score")

plt.title("Ensemble mean average F1 score")

plt.legend(loc='lower right')
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
plt.axis([0, 33, 0.70, 0.78])

plt.xlabel(u"Total base learners")
plt.ylabel(u"Average F1 score")

plt.title("Ensemble best average F1 score")

plt.legend(loc='lower right')
plt.show()

'''







'''
#plotar o desempenho medio de acordo com o tamanho do treino
#ensemble features iguais 9 learners

tamanho_treino = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
acc_bagging = np.array([0.7335, 0.7435, 0.7483, 0.7519, 0.7550, 0.757, 0.7584, 0.76, 0.7618, 0.7624, 0.7643, 0.7652, 0.7663, 0.7649, 0.7653, 0.7650, 0.7674, 0.7677, 0.7676])
acc_boost = np.array([0.7228, 0.7315, 0.7385, 0.743, 0.7473, 0.7492, 0.7535, 0.7546, 0.7564, 0.7590, 0.7583, 0.7603, 0.7621, 0.7644, 0.7650, 0.7656, 0.7681, 0.7677, 0.7666])

plt.plot(tamanho_treino, acc_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(tamanho_treino, acc_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 1, 0.70, 0.78])

plt.xlabel(u"Training sample")
plt.ylabel(u"Accuracy")

plt.title("Ensemble mean accuracy")

plt.legend(loc='lower right')
plt.show()

'''



'''
#plotar melhor desempenho de acordo com o tamanho do treino
#ensemble features iguais 9 learners

tamanho_treino = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
melhor_acc_bagging = np.array([0.738, 0.746, 0.750, 0.754, 0.757, 0.759, 0.760, 0.761, 0.763, 0.763, 0.765, 0.766, 0.767, 0.766, 0.766, 0.765, 0.768, 0.768, 0.768])
melhor_acc_boost = np.array([0.734, 0.741, 0.748, 0.752, 0.756, 0.756, 0.760, 0.760, 0.761, 0.763, 0.763, 0.764, 0.765, 0.766, 0.767, 0.766, 0.769, 0.768, 0.767])

plt.plot(tamanho_treino, melhor_acc_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(tamanho_treino, melhor_acc_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 1, 0.70, 0.78])

plt.xlabel(u"Training sample")
plt.ylabel(u"Accuracy")

plt.title("Ensemble best accuracy")

plt.legend(loc='lower right')
plt.show()

'''





'''
#plotar f1 medio de acordo com o tamanho do treino
#ensemble features iguais 9 learners

tamanho_treino = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
f1_bagging = np.array([0.7057, 0.7158, 0.7215, 0.7267, 0.7302, 0.7334, 0.7358, 0.7378, 0.7409, 0.7421, 0.7445, 0.7459, 0.7481, 0.7472, 0.7479, 0.7481, 0.7507, 0.7516, 0.7520])
f1_boost = np.array([0.7055, 0.7172, 0.7244, 0.7299, 0.7352, 0.737, 0.7417, 0.7431, 0.7453, 0.7473, 0.7465, 0.7488, 0.7496, 0.7509, 0.7504, 0.7503, 0.7523, 0.7517, 0.7508])

plt.plot(tamanho_treino, f1_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(tamanho_treino, f1_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 1, 0.7, 0.78])

plt.xlabel(u"Training sample")
plt.ylabel(u"Average F1 score")

plt.title("Ensemble mean average F1 score")

plt.legend(loc='lower right')
plt.show()

'''


'''
#plotar melhor f1 de acordo com o tamanho do treino
#ensemble features iguais 9 learners

tamanho_treino = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
melhor_f1_bagging = np.array([0.712, 0.723, 0.727, 0.731, 0.735, 0.737, 0.739, 0.741, 0.745, 0.745, 0.748, 0.750, 0.751, 0.750, 0.750, 0.751, 0.754, 0.755, 0.755])
melhor_f1_boost = np.array([0.714, 0.723, 0.730, 0.735, 0.739, 0.739, 0.744, 0.745, 0.747, 0.749, 0.748, 0.751, 0.753, 0.753, 0.753, 0.753, 0.755, 0.755, 0.754])

plt.plot(tamanho_treino, melhor_f1_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(tamanho_treino, melhor_f1_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 1, 0.7, 0.78])

plt.xlabel(u"Training sample")
plt.ylabel(u"Average F1 score")

plt.title("Ensemble best average F1 score")

plt.legend(loc='lower right')
plt.show()

'''


'''
#plotar performance combinando features
#ensemble 9 learners
#5 features fixas

acc_bagging = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7375, 0.7474, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7371, 0.7445, 0.7508, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7354, 0.7457, 0.7533, 0.7559, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7330, 0.7447, 0.7514, 0.7554, 0.7619, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.7169, 0.7333, 0.7449, 0.7514, 0.7530, 0.7600, 0.7619, 0.7614, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.7166, 0.7314, 0.7418, 0.7489, 0.7527, 0.7599, 0.7602, 0.7579, 0.7613, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.7160, 0.7321, 0.7427, 0.7484, 0.7519, 0.7593, 0.7594, 0.7620, 0.7618, 0.7638, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.7190, 0.7345, 0.7424, 0.7475, 0.7521, 0.7559, 0.7596, 0.7611, 0.7626, 0.7618, 0.7625, 0.0, 0.0, 0.0, 0.0],
[0.7207, 0.7346, 0.7437, 0.7478, 0.7515, 0.7573, 0.7573, 0.7582, 0.7610, 0.7622, 0.7600, 0.7585, 0.0, 0.0, 0.0],
[0.7164, 0.7343, 0.7406, 0.7463, 0.7495, 0.7565, 0.7564, 0.7570, 0.7589, 0.7585, 0.7587, 0.7579, 0.7594, 0.0, 0.0],
[0.7141, 0.7309, 0.7393, 0.7444, 0.7487, 0.7518, 0.7549, 0.7574, 0.7551, 0.7547, 0.7541, 0.7574, 0.7562, 0.7569, 0.0]])


fig = plt.figure()
mapa = fig.add_subplot(111)

norm = Normalize(vmin = 0.6900, vmax = 0.7700, clip = False)

cmap = cm.get_cmap(None, None)
cmap.set_under('w')

ticks_x = [x+1 for x in range(15)]
ticks_y = [y+1 for y in range(15)]
plt.xticks(ticks_x)
plt.yticks(ticks_y)

mapa.grid(True, which='both')
imagem = mapa.imshow(acc_bagging, interpolation='none', cmap=cmap, norm=norm, extent=[0.5,15.5,15.5,0.5])
plt.colorbar(imagem)

plt.xlabel("Selected features")
plt.ylabel("Available features")
plt.title("Bagging mean accuracy heatmap")

plt.show()

'''


'''
#plotar performance combinando features
#ensemble 9 learners
#5 features fixas

acc_boost = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7286, 0.7398, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7286, 0.7384, 0.7488, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7258, 0.7353, 0.7458, 0.7495, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7247, 0.7374, 0.7430, 0.7470, 0.7524, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.7012, 0.7207, 0.7341, 0.7446, 0.7477, 0.7506, 0.7514, 0.7560, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.6982, 0.7189, 0.7316, 0.7397, 0.7432, 0.7536, 0.7506, 0.7520, 0.7560, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.6967, 0.7201, 0.7331, 0.7417, 0.7438, 0.7511, 0.7510, 0.7564, 0.7546, 0.7584, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.7011, 0.7223, 0.7337, 0.7403, 0.7456, 0.7498, 0.7542, 0.7548, 0.7593, 0.7580, 0.7580, 0.0, 0.0, 0.0, 0.0],
[0.6992, 0.7205, 0.7339, 0.7395, 0.7423, 0.7470, 0.7462, 0.7552, 0.7531, 0.7574, 0.7568, 0.7577, 0.0, 0.0, 0.0],
[0.7006, 0.7196, 0.7282, 0.7398, 0.7439, 0.7479, 0.7455, 0.7497, 0.7526, 0.7536, 0.7552, 0.7545, 0.7567, 0.0, 0.0],
[0.6975, 0.7204, 0.7288, 0.7371, 0.7396, 0.7447, 0.7465, 0.7480, 0.7490, 0.7495, 0.7497, 0.7511, 0.7533, 0.7555, 0.0]])


fig = plt.figure()
mapa = fig.add_subplot(111)

norm = Normalize(vmin = 0.6900, vmax = 0.7700, clip = False)

cmap = cm.get_cmap(None, None)
cmap.set_under('w')

ticks_x = [x+1 for x in range(15)]
ticks_y = [y+1 for y in range(15)]
plt.xticks(ticks_x)
plt.yticks(ticks_y)

mapa.grid(True, which='both')
imagem = mapa.imshow(acc_boost, interpolation='none', cmap=cmap, norm=norm, extent=[0.5,15.5,15.5,0.5])
plt.colorbar(imagem)

plt.xlabel("Selected features")
plt.ylabel("Available features")
plt.title("Boost mean accuracy heatmap")

plt.show()

'''


'''
#plotar melhor performance combinando features
#ensemble 9 learners
#5 features fixas

melhor_acc_bagging = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.750, 0.753, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.748, 0.752, 0.757, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.747, 0.752, 0.757, 0.761, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.744, 0.754, 0.757, 0.760, 0.764, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.736, 0.746, 0.752, 0.757, 0.759, 0.764, 0.764, 0.763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.734, 0.744, 0.750, 0.757, 0.757, 0.763, 0.762, 0.759, 0.763, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.735, 0.744, 0.752, 0.755, 0.757, 0.763, 0.762, 0.764, 0.763, 0.765, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.736, 0.747, 0.751, 0.755, 0.757, 0.761, 0.761, 0.764, 0.764, 0.764, 0.764, 0.0, 0.0, 0.0, 0.0],
[0.736, 0.746, 0.751, 0.754, 0.758, 0.761, 0.760, 0.760, 0.762, 0.764, 0.762, 0.759, 0.0, 0.0, 0.0],
[0.733, 0.746, 0.750, 0.752, 0.755, 0.762, 0.759, 0.759, 0.760, 0.760, 0.760, 0.759, 0.761, 0.0, 0.0],
[0.731, 0.743, 0.746, 0.751, 0.754, 0.756, 0.758, 0.759, 0.756, 0.757, 0.755, 0.759, 0.758, 0.758, 0.0]])


fig = plt.figure()
mapa = fig.add_subplot(111)

norm = Normalize(vmin = 0.6900, vmax = 0.7700, clip = False)

cmap = cm.get_cmap(None, None)
cmap.set_under('w')

ticks_x = [x+1 for x in range(15)]
ticks_y = [y+1 for y in range(15)]
plt.xticks(ticks_x)
plt.yticks(ticks_y)

mapa.grid(True, which='both')
imagem = mapa.imshow(melhor_acc_bagging, interpolation='none', cmap=cmap, norm=norm, extent=[0.5,15.5,15.5,0.5])
plt.colorbar(imagem)

plt.xlabel("Selected features")
plt.ylabel("Available features")
plt.title("Bagging best accuracy heatmap")

plt.show()

'''


'''
#plotar melhor performance combinando features
#ensemble 9 learners
#5 features fixas

melhor_acc_boost = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.745, 0.748, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.744, 0.750, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.743, 0.748, 0.755, 0.758, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.743, 0.751, 0.752, 0.754, 0.760, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.735, 0.743, 0.748, 0.755, 0.756, 0.759, 0.759, 0.761, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.732, 0.739, 0.747, 0.751, 0.752, 0.762, 0.760, 0.758, 0.759, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.729, 0.740, 0.747, 0.753, 0.756, 0.760, 0.761, 0.762, 0.762, 0.764, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.732, 0.741, 0.747, 0.753, 0.756, 0.758, 0.761, 0.763, 0.765, 0.764, 0.763, 0.0, 0.0, 0.0, 0.0],
[0.728, 0.742, 0.747, 0.752, 0.754, 0.757, 0.756, 0.761, 0.759, 0.763, 0.763, 0.761, 0.0, 0.0, 0.0],
[0.729, 0.738, 0.746, 0.751, 0.753, 0.757, 0.756, 0.756, 0.759, 0.759, 0.760, 0.759, 0.761, 0.0, 0.0],
[0.729, 0.737, 0.743, 0.748, 0.750, 0.752, 0.753, 0.755, 0.755, 0.753, 0.753, 0.756, 0.757, 0.759, 0.0]])


fig = plt.figure()
mapa = fig.add_subplot(111)

norm = Normalize(vmin = 0.6900, vmax = 0.7700, clip = False)

cmap = cm.get_cmap(None, None)
cmap.set_under('w')

ticks_x = [x+1 for x in range(15)]
ticks_y = [y+1 for y in range(15)]
plt.xticks(ticks_x)
plt.yticks(ticks_y)

mapa.grid(True, which='both')
imagem = mapa.imshow(melhor_acc_boost, interpolation='none', cmap=cmap, norm=norm, extent=[0.5,15.5,15.5,0.5])
plt.colorbar(imagem)

plt.xlabel("Selected features")
plt.ylabel("Available features")
plt.title("Boost best accuracy heatmap")

plt.show()

'''



'''
#plotar f1 combinando features
#ensemble 9 learners
#5 features fixas

f1_bagging = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7265, 0.7346, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7272, 0.7323, 0.7376, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7260, 0.7334, 0.7401, 0.7419, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7231, 0.7326, 0.7380, 0.7407, 0.7453, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.7116, 0.7245, 0.7333, 0.7384, 0.7387, 0.7441, 0.7447, 0.7427, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.7108, 0.7230, 0.7307, 0.7362, 0.7391, 0.7437, 0.7427, 0.7392, 0.7415, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.7106, 0.7234, 0.7317, 0.7355, 0.7376, 0.7432, 0.7412, 0.7427, 0.7408, 0.7418, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.7137, 0.7260, 0.7311, 0.7337, 0.7369, 0.7383, 0.7399, 0.7405, 0.7403, 0.7380, 0.7376, 0.0, 0.0, 0.0, 0.0],
[0.7145, 0.7256, 0.7322, 0.7344, 0.7372, 0.7403, 0.7376, 0.7375, 0.7391, 0.7398, 0.7344, 0.7320, 0.0, 0.0, 0.0],
[0.7206, 0.7252, 0.7285, 0.7327, 0.7353, 0.7397, 0.7374, 0.7367, 0.7371, 0.7353, 0.7329, 0.7309, 0.7294, 0.0, 0.0],
[0.7090, 0.7216, 0.7283, 0.7312, 0.7340, 0.7346, 0.7361, 0.7372, 0.7328, 0.7314, 0.7273, 0.7298, 0.7247, 0.7247, 0.0]])


fig = plt.figure()
mapa = fig.add_subplot(111)

norm = Normalize(vmin = 0.6900, vmax = 0.7700, clip = False)

cmap = cm.get_cmap(None, None)
cmap.set_under('w')

ticks_x = [x+1 for x in range(15)]
ticks_y = [y+1 for y in range(15)]
plt.xticks(ticks_x)
plt.yticks(ticks_y)

mapa.grid(True, which='both')
imagem = mapa.imshow(f1_bagging, interpolation='none', cmap=cmap, norm=norm, extent=[0.5,15.5,15.5,0.5])
plt.colorbar(imagem)

plt.xlabel("Selected features")
plt.ylabel("Available features")
plt.title("Bagging mean average F1 score heatmap")

plt.show()

'''


'''
#plotar f1 combinando features
#ensemble 9 learners
#5 features fixas

f1_boost = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7179, 0.7284, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7189, 0.7277, 0.7382, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7171, 0.7247, 0.7352, 0.7386, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.7156, 0.7264, 0.7314, 0.7351, 0.7408, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.6966, 0.7128, 0.7235, 0.7337, 0.7364, 0.7393, 0.7402, 0.7440, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.6937, 0.7116, 0.7223, 0.7291, 0.7320, 0.7418, 0.7393, 0.7393, 0.7443, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.6921, 0.7119, 0.7232, 0.7310, 0.7320, 0.7388, 0.7388, 0.7436, 0.7417, 0.7453, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.6961, 0.7139, 0.7232, 0.7293, 0.7338, 0.7373, 0.7414, 0.7421, 0.7467, 0.7444, 0.7440, 0.0, 0.0, 0.0, 0.0],
[0.6937, 0.7119, 0.7234, 0.7285, 0.7303, 0.7352, 0.7332, 0.7416, 0.7399, 0.7446, 0.7426, 0.7437, 0.0, 0.0, 0.0],
[0.6945, 0.7114, 0.7170, 0.7286, 0.7317, 0.7354, 0.7325, 0.7357, 0.7383, 0.7396, 0.7407, 0.7394, 0.7421, 0.0, 0.0],
[0.6919, 0.7116, 0.7180, 0.7250, 0.7279, 0.7319, 0.7336, 0.7342, 0.7347, 0.7349, 0.7349, 0.7363, 0.7384, 0.7396, 0.0]])


fig = plt.figure()
mapa = fig.add_subplot(111)

norm = Normalize(vmin = 0.6900, vmax = 0.7700, clip = False)

cmap = cm.get_cmap(None, None)
cmap.set_under('w')

ticks_x = [x+1 for x in range(15)]
ticks_y = [y+1 for y in range(15)]
plt.xticks(ticks_x)
plt.yticks(ticks_y)

mapa.grid(True, which='both')
imagem = mapa.imshow(f1_boost, interpolation='none', cmap=cmap, norm=norm, extent=[0.5,15.5,15.5,0.5])
plt.colorbar(imagem)

plt.xlabel("Selected features")
plt.ylabel("Available features")
plt.title("Boost mean average F1 score heatmap")

plt.show()

'''


'''
#plotar melhor f1 combinando features
#ensemble 9 learners
#5 features fixas

melhor_f1_bagging = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.731, 0.737, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.730, 0.734, 0.740, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.730, 0.735, 0.742, 0.745, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.727, 0.736, 0.740, 0.743, 0.749, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.718, 0.730, 0.737, 0.741, 0.741, 0.746, 0.747, 0.746, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.717, 0.726, 0.733, 0.738, 0.742, 0.746, 0.746, 0.742, 0.746, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.716, 0.728, 0.734, 0.738, 0.740, 0.746, 0.744, 0.745, 0.744, 0.746, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.721, 0.730, 0.734, 0.736, 0.739, 0.741, 0.743, 0.743, 0.744, 0.742, 0.741, 0.0, 0.0, 0.0, 0.0],
[0.720, 0.729, 0.735, 0.737, 0.739, 0.742, 0.741, 0.741, 0.743, 0.744, 0.739, 0.736, 0.0, 0.0, 0.0],
[0.718, 0.729, 0.731, 0.735, 0.739, 0.741, 0.741, 0.741, 0.741, 0.739, 0.737, 0.735, 0.733, 0.0, 0.0],
[0.715, 0.725, 0.731, 0.734, 0.736, 0.738, 0.740, 0.741, 0.737, 0.736, 0.731, 0.733, 0.729, 0.730, 0.0]])


fig = plt.figure()
mapa = fig.add_subplot(111)

norm = Normalize(vmin = 0.6900, vmax = 0.7700, clip = False)

cmap = cm.get_cmap(None, None)
cmap.set_under('w')

ticks_x = [x+1 for x in range(15)]
ticks_y = [y+1 for y in range(15)]
plt.xticks(ticks_x)
plt.yticks(ticks_y)

mapa.grid(True, which='both')
imagem = mapa.imshow(melhor_f1_bagging, interpolation='none', cmap=cmap, norm=norm, extent=[0.5,15.5,15.5,0.5])
plt.colorbar(imagem)

plt.xlabel("Selected features")
plt.ylabel("Available features")
plt.title("Bagging best average F1 score heatmap")

plt.show()

'''


'''
#plotar melhor f1 combinando features
#ensemble 9 learners
#5 features fixas

melhor_f1_boost = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.724, 0.733, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.725, 0.730, 0.742, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.724, 0.729, 0.739, 0.742, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.0, 0.723, 0.732, 0.735, 0.739, 0.743, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.712, 0.722, 0.730, 0.738, 0.741, 0.744, 0.743, 0.746, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.707, 0.720, 0.729, 0.732, 0.735, 0.745, 0.742, 0.742, 0.747, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.705, 0.720, 0.728, 0.735, 0.734, 0.742, 0.741, 0.746, 0.744, 0.747, 0.0, 0.0, 0.0, 0.0, 0.0],
[0.708, 0.719, 0.727, 0.732, 0.737, 0.740, 0.744, 0.746, 0.750, 0.747, 0.746, 0.0, 0.0, 0.0, 0.0],
[0.707, 0.719, 0.728, 0.733, 0.734, 0.739, 0.735, 0.744, 0.742, 0.747, 0.744, 0.746, 0.0, 0.0, 0.0],
[0.706, 0.720, 0.722, 0.731, 0.735, 0.738, 0.734, 0.738, 0.741, 0.742, 0.743, 0.742, 0.744, 0.0, 0.0],
[0.703, 0.718, 0.723, 0.728, 0.731, 0.735, 0.738, 0.737, 0.738, 0.738, 0.738, 0.739, 0.741, 0.742, 0.0]])


fig = plt.figure()
mapa = fig.add_subplot(111)

norm = Normalize(vmin = 0.6900, vmax = 0.7700, clip = False)

cmap = cm.get_cmap(None, None)
cmap.set_under('w')

ticks_x = [x+1 for x in range(15)]
ticks_y = [y+1 for y in range(15)]
plt.xticks(ticks_x)
plt.yticks(ticks_y)

mapa.grid(True, which='both')
imagem = mapa.imshow(melhor_f1_boost, interpolation='none', cmap=cmap, norm=norm, extent=[0.5,15.5,15.5,0.5])
plt.colorbar(imagem)

plt.xlabel("Selected features")
plt.ylabel("Available features")
plt.title("Boost best average F1 score heatmap")

plt.show()

'''


######
# a05_b10_c04
######



'''
#plotar o desempenho medio de acordo com o tamanho do treino
#ensemble a05_b10_c04 9 learners

tamanho_treino = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
acc_bagging = np.array([0.7320, 0.7453, 0.7502, 0.7539, 0.7559, 0.7593, 0.7590, 0.7590, 0.7634, 0.7638, 0.7644, 0.7642, 0.7666, 0.7662, 0.7650, 0.7656, 0.7676, 0.7669, 0.7690])
acc_boost = np.array([0.7213, 0.7316, 0.7394, 0.7418, 0.7467, 0.7508, 0.7503, 0.7519, 0.7532, 0.7584, 0.7595, 0.7587, 0.7608, 0.7634, 0.7623, 0.7650, 0.7666, 0.7661, 0.7665])

plt.plot(tamanho_treino, acc_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(tamanho_treino, acc_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 1, 0.70, 0.78])

plt.xlabel(u"Training sample")
plt.ylabel(u"Accuracy")

plt.title("Ensemble mean accuracy")

plt.legend(loc='lower right')
plt.show()
'''




'''
#plotar melhor desempenho de acordo com o tamanho do treino
#ensemble a05_b10_c04 9 learners

tamanho_treino = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
melhor_acc_bagging = np.array([0.737, 0.748, 0.753, 0.756, 0.758, 0.762, 0.761, 0.761, 0.765, 0.765, 0.766, 0.765, 0.767, 0.767, 0.766, 0.767, 0.769, 0.768, 0.770])
melhor_acc_boost = np.array([0.731, 0.742, 0.750, 0.752, 0.756, 0.760, 0.759, 0.760, 0.758, 0.764, 0.765, 0.764, 0.766, 0.767, 0.765, 0.767, 0.769, 0.767, 0.769])

plt.plot(tamanho_treino, melhor_acc_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(tamanho_treino, melhor_acc_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 1, 0.70, 0.78])

plt.xlabel(u"Training sample")
plt.ylabel(u"Accuracy")

plt.title("Ensemble best accuracy")

plt.legend(loc='lower right')
plt.show()
'''






'''
#plotar f1 medio de acordo com o tamanho do treino
#ensemble a05_b10_c04 9 learners

tamanho_treino = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
f1_bagging = np.array([0.7037, 0.7176, 0.7227, 0.7276, 0.7297, 0.7347, 0.7351, 0.7357, 0.7411, 0.7418, 0.7435, 0.7436, 0.7464, 0.7464, 0.7460, 0.7468, 0.7490, 0.7491, 0.7515])
f1_boost = np.array([0.7039, 0.7173, 0.7252, 0.7287, 0.7335, 0.7379, 0.7379, 0.7392, 0.7407, 0.7453, 0.7465, 0.7460, 0.7470, 0.7487, 0.7463, 0.7481, 0.7492, 0.7485, 0.7493])

plt.plot(tamanho_treino, f1_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(tamanho_treino, f1_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 1, 0.7, 0.78])

plt.xlabel(u"Training sample")
plt.ylabel(u"Average F1 score")

plt.title("Ensemble mean average F1 score")

plt.legend(loc='lower right')
plt.show()
'''



'''
#plotar melhor f1 de acordo com o tamanho do treino
#ensemble a05_b10_c04 9 learners

tamanho_treino = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
melhor_f1_bagging = np.array([0.710, 0.724, 0.729, 0.733, 0.734, 0.739, 0.739, 0.739, 0.745, 0.746, 0.747, 0.747, 0.749, 0.750, 0.749, 0.750, 0.752, 0.752, 0.755])
melhor_f1_boost = np.array([0.711, 0.724, 0.731, 0.734, 0.738, 0.741, 0.741, 0.742, 0.743, 0.747, 0.749, 0.748, 0.749, 0.751, 0.749, 0.751, 0.751, 0.752, 0.752])

plt.plot(tamanho_treino, melhor_f1_bagging, 'ro', linestyle='-', label="Bagging")
plt.plot(tamanho_treino, melhor_f1_boost, 'b^', linestyle='-', label="Boost")
plt.axis([0, 1, 0.7, 0.78])

plt.xlabel(u"Training sample")
plt.ylabel(u"Average F1 score")

plt.title("Ensemble best average F1 score")

plt.legend(loc='lower right')
plt.show()
'''

