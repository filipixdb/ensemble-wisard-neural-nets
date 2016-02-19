'''
@author: filipi
'''
import pprint

'''
Testar os classificadores no dataset german credit
'''


import time
import sys
sys.path.append("../")

import wann.util as util
import ensemble.composition as compo
import ensemble.classifiers as e_clss
import data_process.preprocess as p_prcs
import ensemble.partition as prtt


def main(nmbr_neurons_list=[16], data=None, bagging=None, percent=None):

    # escolher parametros
    
    n_folds = 8
    nmbr_neurons_list = [8, 10, 16, 20]
    bagging = None
    percent = 0.5
    tipos_classi = 3
    features = 13
    qnt_class_por_tipo = 4

    print "Neuronios: ", nmbr_neurons_list
    print "Folds: ", n_folds
    print "Bagging: ", bagging
    print "Percent: ", percent
    
    # iniciar o vetor de selected features
    selected_features = p_prcs.sorteia_features(qnt_tipos_neuro=len(nmbr_neurons_list),
                                                tipos_classi=tipos_classi, qnt_por_classi=qnt_class_por_tipo,
                                                total_feat=20, qnt_feat=features)
   
    # ler os dados
    if data is not None:
        data = "files/"+data
    else:
        data = "files/encoded_german.data"
    print "DATA: ", data
    data_names_dict = p_prcs.le_features(data, selected_features)
    
    
    # Cria os classificadores
    classificadores, _, labels = e_clss.cria_classificadores(selected_features, nmbr_neurons_list, tipos_classi)


    # Cria os folds
    _, _, fold_data_dict = prtt.cria_folds(selected_features, data_names_dict, nmbr_neurons_list, n_folds = n_folds)

    #pprint.pprint("Fold Data Dict[0]:")
    #pprint.pprint(fold_data_dict[str(0)])
    '''
    fold_data_dict possui:
    pra cada classificador
      pra cada fold
        list features, classe
    '''
    
    

#    Inicializa e treina os classificadores, inicializa as matrizes e os cv_gen
    confusion_mat, times, cv_gen = e_clss.inicializa_classificadores(classificadores, fold_data_dict, bagging, percent)
    confusion_mat_ensemble = util.ConfusionMatrix()
    confusion_mat_ensemble_com_pesos = util.ConfusionMatrix()
    
    
    # iterar nos folds
    for _ in range(n_folds):
        
        
        for _, _, _, _, chave in classificadores:
            _ = cv_gen[chave].next()
        
        
        # persistir os votos e scores de cada classificador em cada fold
        votos = {}
        scores = {}
        classes = []
        classes_preenchidas = False
        
        # construir matrizes de confusao para os folds (pro ensemble com pesos por performance)
        confusion_mat_fold = e_clss.inicializa_matrizes_de_confusao(classificadores)
        pesos_classif = []
                
        # iterar nos classificadores
        for _, _, _, _, chave in classificadores:
            time_alfa = time.time()
            respostas = []
            pesos = []
            
            
            for ax in cv_gen[chave]: # iterar nas instancias
                if ax is None:
                    break
                
                _, answers, clss = ax # ax eh um monte de tuplas (observation, respostaClassificador, respostaReal)
                rank = util.ranked(answers) # rank recebe varias tuplas ordenadas ('classe', somaRespostasDosDiscriminadores)
                # top_score recebe a maior soma dos discriminadores
                try:
                    top_score = len(rank[0][1])
                except TypeError:
                    top_score = rank[0][1]
                
                # coloca na matriz de confusao a resposta e o valor da soma dos discriminadores
                confusion_mat[chave].add(clss, rank[0][0], top_score)
                confusion_mat_fold[chave].add(clss, rank[0][0], top_score)# essa sera usada pro ensemble com peso por performance
                respostas.append(rank[0][0])
                pesos.append(top_score)
                
                # guardar a classificacao correta
                if not classes_preenchidas :
                    classes.append(clss)
                
            
            
            votos[chave] = respostas
            scores[chave] = pesos
            times[chave] += (time.time() - time_alfa)
            classes_preenchidas = True
            
            # definir o peso do classificador = performance
            _, acuracia = confusion_mat[chave].stats()[2]
            pesos_classif.append(acuracia)
        
                
        # montar o array com os votos dos classificadores
        votos_transposta = []
        for i in range(len(classes)): # para cada instancia
            row = []
            #for j in range(len(chaves_classificadores)): # para cada classificador
            for _, _, _, _, chave in classificadores:# Mais correto usar o chave, sempre tentar iterar as coisas com uma mesma ordenacao
                row.append(votos[chave][i])
            votos_transposta.append(row)
        
        
        
        # fazer o ensemble dos classificadores
        combinador = compo.VotingAggregator(votos_transposta, len(votos_transposta), len(votos_transposta[0]), 2)
        combinador.predict()
        votos_combinados = combinador.combined_votes
        
        # fazer o ensemble com pesos de acordo com a performance
        combinador_com_pesos = compo.VotingAggregator(votos_transposta, len(votos_transposta),
                                                      len(votos_transposta[0]), 2, vote='weightedClassifiers',
                                                      weights=pesos_classif)
        combinador_com_pesos.predict()
        votos_combinados_com_pesos = combinador_com_pesos.combined_votes
        
        #print pesos_classif
             
        
        # avaliar o ensemble sem e com peso
        for i in range(len(votos_combinados)):
            confusion_mat_ensemble.add(int(classes[i]), votos_combinados[i], 0)
            confusion_mat_ensemble_com_pesos.add(int(classes[i]), votos_combinados_com_pesos[i], 0)
    
    
    #for i in range(len(confusion_mat)):
    for _, _, _, _, chave in classificadores:
        print confusion_mat[chave]
        print confusion_mat[chave].stats()
        print labels[chave], 'time:',  times[chave]
        
    print confusion_mat_ensemble
    print confusion_mat_ensemble.stats()
    print 'Ensemble (majority)'
    
    print confusion_mat_ensemble_com_pesos
    print confusion_mat_ensemble_com_pesos.stats()
    print 'Ensemble (pesos)'
    
    


if __name__ == '__main__':
    main(*sys.argv[1:])
#    main()

