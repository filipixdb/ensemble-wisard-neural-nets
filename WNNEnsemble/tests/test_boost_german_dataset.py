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

    
    # ler informacoes
    arquivo = 'encoded_german_20_bits'
    n_classes, tam_features, n_params, list_n_folds = frdr.le_informacoes('../tests/files/'+arquivo+'.info')
    
    
    # le as entradas
    data = frdr.le_entradas('../tests/files/'+arquivo+'_TEMP.data', tam_features)
    
    
    for x in range(n_params):
        print "\nParams ", x, " ========================================"
        # ler configs dos single e base learners
        configs_single_learners, configs_base_learners = frdr.le_parametros('../tests/files/'+arquivo+'.params'+str(x))
        
        # aqui criar as matrizes de confusao dos dois ensembles dos dois algoritmos
        matrizes_ensemble = e_clss.cria_matriz_confusao_geral_ensemble()
        
    
        for n_folds in list_n_folds:
            print "\nFolds = ", n_folds
            # criar o dataset e folds
            dataset = DataSet(data, tam_features, n_folds, n_classes, nome='german dataset')
            
            print "\n  Bagging"
            executa_algoritmo("Bagging", dataset, n_folds, configs_single_learners, configs_base_learners, matrizes_ensemble['Bagging'], 0.3, False)
            
            print "\n  AdaBoost"
            executa_algoritmo("AdaBoost", dataset, n_folds, [], configs_base_learners, matrizes_ensemble['AdaBoost'], 0.5, False)



def executa_algoritmo(algoritmo, dataset, n_folds, configs_single_learners, configs_base_learners, matrizes_ensemble, amostragem, repeticao):
    # criar os single learners
    single_learners = e_clss.cria_learners(configs_single_learners, n_folds, dataset.tam_features, mapping_igual=False)
    # criar os base learners
    base_learners = e_clss.cria_learners(configs_base_learners, n_folds, dataset.tam_features)
    
    # cria os ensembles
    ensembles = []
    if len(base_learners) > 0:
        ensembles.append(Ensemble('majority', n_folds, matrizes_ensemble['majority']))
        ensembles.append(Ensemble('weightedClassifiers', n_folds, matrizes_ensemble['weightedClassifiers']))
        ensembles.append(Ensemble('majority', n_folds, matrizes_ensemble['majority'], com_confiancas=True))
        ensembles.append(Ensemble('weightedClassifiers', n_folds, matrizes_ensemble['weightedClassifiers'], com_confiancas=True))
    

    if algoritmo=='AdaBoost':
        alg = AdaBoost(dataset, base_learners, single_learners, ensembles, amostragem, repeticao)
    elif algoritmo=='Bagging':
        alg = Bagging(dataset, base_learners, single_learners, ensembles, amostragem, repeticao)

    alg.executa_folds()



if __name__ == '__main__':
    main()

