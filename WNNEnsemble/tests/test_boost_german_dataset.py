'''
Testar os classificadores no dataset german credit
'''

import sys
import random
sys.path.append("../")

import ensemble.classifiers as e_clss
import data_process.file_reader as frdr
from data_model.dataset import DataSet
from ensemble.classifiers import Ensemble
from ensemble.boost import AdaBoost, Bagging


def main():
    
    # ler informacoes
    arquivo = 'encoded_german_20_bits'
    
    # le arquivo de informacoes
    n_classes, tam_features, parametrizacoes = frdr.le_informacoes('../tests/files/'+arquivo+'.info')
    #n_classes, tam_features, n_params, list_n_folds = frdr.le_informacoes('../tests/files/'+arquivo+'.info')
    
    # le as entradas
    data = frdr.le_entradas('../tests/files/'+arquivo+'_TEMP.data', tam_features, shuffle=True)
    
    
    for parametrizacao in parametrizacoes:
        print "\nParametrizacao: ", parametrizacao
        
        # ler configs dos single e base learners
        #configs_single_learners, configs_base_learners = frdr.le_parametros('../tests/files/'+arquivo+'.params'+str(x))
        n_folds, n_execucoes, tamanho_treino_bagging, tamanho_treino_boost, com_repeticao_bagging, com_repeticao_boost, mesmo_mapping_bagging, mesmo_mapping_boost, configs_single_learners, configs_base_learners, n_base_learners = frdr.le_parametros('../tests/files/'+'param_'+parametrizacao)
        
        # aqui criar as matrizes de confusao dos dois ensembles dos dois algoritmos
        matrizes_ensemble = e_clss.cria_matriz_confusao_geral_ensemble()
        
    
        for execucao in xrange(n_execucoes):
            print "\n  execucao: ", execucao
            
            # embaralhar instancias
            data = random.sample(data, len(data))

            # sortear base learners
            if n_base_learners == None:
                configs_base_learners_escolhidos = configs_base_learners
            else:
                configs_base_learners_escolhidos = [ configs_base_learners[i] for i in sorted(random.sample(xrange(len(configs_base_learners)), n_base_learners)) ]
    
    
            # criar o dataset e folds
            dataset = DataSet(data, tam_features, n_folds, n_classes, nome='german dataset')
            
            print "\n    Bagging"
            executa_algoritmo("Bagging", dataset, n_folds, configs_single_learners, configs_base_learners_escolhidos, matrizes_ensemble['Bagging'], tamanho_treino_bagging, com_repeticao_bagging, mesmo_mapping_bagging)
            
            print "\n    AdaBoost"
            executa_algoritmo("AdaBoost", dataset, n_folds, [], configs_base_learners_escolhidos, matrizes_ensemble['AdaBoost'], tamanho_treino_boost, com_repeticao_boost, mesmo_mapping_boost)

    
    '''
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
    '''


def executa_algoritmo(algoritmo, dataset, n_folds, configs_single_learners, configs_base_learners, matrizes_ensemble, amostragem, repeticao, mapping_igual):
    # criar os single learners
    single_learners = e_clss.cria_learners(configs_single_learners, n_folds, dataset.tam_features, mapping_igual)
    # criar os base learners
    base_learners = e_clss.cria_learners(configs_base_learners, n_folds, dataset.tam_features, mapping_igual)
    
    # cria os ensembles
    ensembles = []
    if len(base_learners) > 0:
        
        tipos_voto = ['majority', 'weightedClassifiers']
        tipos_erro = ['nenhum', 'porcentagem', 'proporcao', 'proporcaoDiferenca', 'ativacao', 'porcentagem+ativacao', 'proporcao+ativacao', 'proporcaoDiferenca+ativacao']
        tipos_intensidade = ['nenhum', 'porcentagem', 'proporcao', 'proporcaoDiferenca', 'ativacao', 'porcentagem+ativacao', 'proporcao+ativacao', 'proporcaoDiferenca+ativacao']
        
        for tipo_voto in tipos_voto:
            for tipo_erro in tipos_erro:
                if tipo_voto == 'majority' and tipo_erro != 'nenhum':
                    continue
                else:
                    for tipo_intensidade in tipos_intensidade:
                        ensembles.append(Ensemble(tipo_voto, n_folds, matrizes_ensemble[tipo_voto+tipo_erro+tipo_intensidade], tipo_erro, tipo_intensidade))
        

    if algoritmo=='AdaBoost':
        alg = AdaBoost(dataset, base_learners, single_learners, ensembles, amostragem, repeticao)
    elif algoritmo=='Bagging':
        alg = Bagging(dataset, base_learners, single_learners, ensembles, amostragem, repeticao)

    alg.executa_folds()


if __name__ == '__main__':
    main()

