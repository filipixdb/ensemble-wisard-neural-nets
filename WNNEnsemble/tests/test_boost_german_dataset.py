'''
@author: filipi
'''
from ensemble.boost import AdaBoost

'''
Testar os classificadores no dataset german credit
'''


import sys
sys.path.append("../")

import ensemble.classifiers as e_clss
import data_process.file_reader as frdr
from data_model.dataset import DataSet
from ensemble.classifiers import Ensemble


def main():

    # escolher parametros
    
    n_folds = 4
    arquivo = 'encoded_german.data'
    
    print "Boosting"
    print "Folds= ", n_folds
    print arquivo
    
    # NOVO
    # informar o tamanho das features
    tam_features = [8] * 20
    # le as entradas
    data = frdr.le_entradas('../tests/files/'+arquivo, tam_features)
    # criar o dataset e folds
    dataset = DataSet(data, tam_features, n_folds, nome='german dataset')
    
    
    
    #NOVO
    # especifica configs classificadores
    configs_base_learners = []

    configs_base_learners.append(('wisard', 'lottery', 8, 'counts', (x for x in xrange(len(tam_features)))))
    #configs_base_learners.append(('wisard', 'lottery', 8, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 8, 'counts', (x for x in xrange(len(tam_features)))))
    #configs_base_learners.append(('wisard', 'wisard', 8, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'lottery', 10, 'counts', (x for x in xrange(len(tam_features)))))
    #configs_base_learners.append(('wisard', 'lottery', 10, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 10, 'counts', (x for x in xrange(len(tam_features)))))
    #configs_base_learners.append(('wisard', 'wisard', 10, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'lottery', 16, 'counts', (x for x in xrange(len(tam_features)))))
    #configs_base_learners.append(('wisard', 'lottery', 16, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 16, 'counts', (x for x in xrange(len(tam_features)))))
    #configs_base_learners.append(('wisard', 'wisard', 16, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'lottery', 20, 'counts', (x for x in xrange(len(tam_features)))))
    #configs_base_learners.append(('wisard', 'lottery', 20, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 20, 'counts', (x for x in xrange(len(tam_features)))))
    #configs_base_learners.append(('wisard', 'wisard', 20, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'lottery', 32, 'counts', (x for x in xrange(len(tam_features)))))
    #configs_base_learners.append(('wisard', 'lottery', 32, 'answers', (x for x in xrange(len(tam_features)))))
    configs_base_learners.append(('wisard', 'wisard', 32, 'counts', (x for x in xrange(len(tam_features)))))
    #configs_base_learners.append(('wisard', 'wisard', 32, 'answers', (x for x in xrange(len(tam_features)))))

    # criar os classificadores
    base_learners = e_clss.cria_base_learners(configs_base_learners, n_folds)
    
    
    #NOVO
    # cria os ensembles
    ensembles = []
    ensembles.append(Ensemble('majority', n_folds))
    ensembles.append(Ensemble('weightedClassifiers', n_folds))
       
    

    #NOVO
    # criar o AdaBoost
    algoritmo = AdaBoost(dataset, base_learners, ensembles, 0.5)
    algoritmo.executa_folds()



if __name__ == '__main__':
    main()

