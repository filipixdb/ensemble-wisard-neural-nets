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
        arq_saida = 'files/param_'+parametrizacao+'_german_dataset.saida'
        with open(arq_saida, 'a') as arq:
            arq.write("\nParametrizacao: "+parametrizacao+'\n')
        
        # ler configs dos single e base learners
        #configs_single_learners, configs_base_learners = frdr.le_parametros('../tests/files/'+arquivo+'.params'+str(x))
        n_folds, n_execucoes, tamanho_treino_bagging, tamanho_treino_boost, com_repeticao_bagging, com_repeticao_boost, mesmo_mapping_bagging, mesmo_mapping_boost, configs_single_learners, configs_base_learners, n_base_learners = frdr.le_parametros('../tests/files/'+'param_'+parametrizacao)
        
        # aqui criar as matrizes de confusao dos dois ensembles dos dois algoritmos
        matrizes_ensemble = e_clss.cria_matriz_confusao_geral_ensemble()
        
    
        for execucao in xrange(n_execucoes):
            if execucao == (n_execucoes-1):
                exibe_resultados = True
            else:
                exibe_resultados = False
               
            print "\n  execucao: ", execucao
            if exibe_resultados:
                with open(arq_saida, 'a') as arq:
                    arq.write("\n  execucao: "+str(execucao)+'\n')
            
            
            # embaralhar instancias
            data = random.sample(data, len(data))

            # sortear base learners
            if n_base_learners == None:
                configs_base_learners_escolhidos = configs_base_learners
            else:
                #TODO: estah sempre ordenando os base learners, no boost isso influencia
                configs_base_learners_escolhidos = [ configs_base_learners[i] for i in sorted(random.sample(xrange(len(configs_base_learners)), n_base_learners)) ]
    
    
            # criar o dataset e folds
            dataset = DataSet(data, tam_features, n_folds, n_classes, nome='files/param_'+parametrizacao+'_german_dataset.saida')
            
            
            print "\n    Bagging"
            if exibe_resultados:
                with open(arq_saida, 'a') as arq:
                    arq.write("\n    Bagging\n")
            
            executa_algoritmo("Bagging", dataset, n_folds, configs_single_learners, configs_base_learners_escolhidos, matrizes_ensemble['Bagging'], tamanho_treino_bagging, com_repeticao_bagging, mesmo_mapping_bagging, exibe_resultados)
            
            
            #TODO: descomentar o codigo, testando sem o boost pro 100 porcento treino
            
            print "\n    AdaBoost"
            if exibe_resultados:
                with open(arq_saida, 'a') as arq:
                    arq.write("\n    AdaBoost\n")
            
            executa_algoritmo("AdaBoost", dataset, n_folds, [], configs_base_learners_escolhidos, matrizes_ensemble['AdaBoost'], tamanho_treino_boost, com_repeticao_boost, mesmo_mapping_boost, exibe_resultados)
            


def executa_algoritmo(algoritmo, dataset, n_folds, configs_single_learners, configs_base_learners, matrizes_ensemble, amostragem, repeticao, mapping_igual, exibe_resultados):
    # criar os single learners
    single_learners = e_clss.cria_learners(configs_single_learners, n_folds, dataset.tam_features, mapping_igual)
    # criar os base learners
    base_learners = e_clss.cria_learners(configs_base_learners, n_folds, dataset.tam_features, mapping_igual)
    
    # cria os ensembles
    ensembles = []
    if len(base_learners) > 0:
        
        #TODO: descomentar, testando 100 porcento treino
        tipos_voto = ['majority', 'weightedClassifiers']
        #tipos_voto = ['majority']#TODO: comentar essa linha
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
        alg = AdaBoost(dataset, base_learners, single_learners, ensembles, amostragem, exibe_resultados, repeticao)
    elif algoritmo=='Bagging':
        alg = Bagging(dataset, base_learners, single_learners, ensembles, amostragem, exibe_resultados, repeticao)

    alg.executa_folds()


if __name__ == '__main__':
    main()

