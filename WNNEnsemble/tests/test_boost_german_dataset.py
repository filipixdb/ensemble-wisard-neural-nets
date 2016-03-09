'''
@author: filipi
'''


'''
Testar os classificadores no dataset german credit
'''

import sys
sys.path.append("../")

import ensemble.classifiers as e_clss
import data_process.file_reader as frdr
from data_model.dataset import DataSet
from ensemble.classifiers import Ensemble
from ensemble.boost import AdaBoost, Bagging


def main():

    # escolher parametros
    
    n_folds = 5
    arquivo = 'encoded_german.data'
    
    print "Folds= ", n_folds
    print arquivo
    
    
    # informar o tamanho das features
    tam_features = [8] * 20
    # le as entradas
    data = frdr.le_entradas('../tests/files/'+arquivo, tam_features)
    # criar o dataset e folds
    dataset = DataSet(data, tam_features, n_folds, nome='german dataset')
    
    
    # configs classificadores
    configs_base_learners = []

    configs_base_learners.append(('wisard', 'wisard', 8, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 10, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 16, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 8, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 10, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 16, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 8, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 10, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 16, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 8, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 10, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 16, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 8, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 10, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 16, 'answers', (x for x in xrange(len(tam_features)))))
#    configs_base_learners.append(('wisard', 'wisard', 10, 'answers', (x for x in xrange(len(tam_features)))))
#    configs_base_learners.append(('wisard', 'wisard', 16, 'answers', (x for x in xrange(len(tam_features)))))
#    configs_base_learners.append(('wisard', 'wisard', 20, 'answers', (x for x in xrange(len(tam_features)))))
#    configs_base_learners.append(('wisard', 'wisard', 32, 'answers', (x for x in xrange(len(tam_features)))))

    # criar os base learners
    base_learners = e_clss.cria_learners(configs_base_learners, n_folds)

    # criar os single learners
    single_learners = e_clss.cria_learners(configs_base_learners, n_folds)

    # cria os ensembles
    ensembles = []
    ensembles.append(Ensemble('majority', n_folds))
    ensembles.append(Ensemble('weightedClassifiers', n_folds))
    

    # criar o AdaBoost
    print "AdaBoost"
    algoritmo = AdaBoost(dataset, base_learners, single_learners, ensembles, 0.5, False)
    algoritmo.executa_folds()

    # criar o Bagging
#    print "Bagging"
#    algoritmo = Bagging(dataset, base_learners, single_learners, ensembles, 0.3, True)
#    algoritmo.executa_folds()



if __name__ == '__main__':
    main()

