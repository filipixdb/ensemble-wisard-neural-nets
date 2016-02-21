'''
Created on 19 de fev de 2016

@author: filipi
'''

'''
fazer os metodos para rodar o treino e o test do boost
'''

import numpy as np
import wann.util as util

class AdaBoost(object):
    
    '''
    Classe responsavel por rodar o algoritmo e armazenar os resultados e metricas
    '''
    
    def __init__(self, dataset, base_learners, ensembles, tam_treino=0.5):
        '''
        receber:
          dataset
          classificadores
          ensembles
          ** talvez as configs de voting
        '''
        self.dataset = dataset
        self.base_learners = base_learners
        self.ensembles = ensembles
        self.tam_treino = tam_treino
        


    def executa_folds(self):
        for fold in self.dataset.folds:
            for base_learner in self.base_learners:
                # samplear
                pesos = fold.retorna_pesos()
                tam_sample = (len(fold.inst_treino) * self.tam_treino)
                sample = np.random.choice(fold.inst_treino, tam_sample, replace=False, p=pesos)
                test = [inst for inst in fold.inst_treino if inst not in sample]
                
                set_corretas = set()
                set_erradas = set()
                
                # itera na parte sampleada do conj treino
                for inst in sample:
                    base_learner.classificador.record(base_learner.encoder(inst.representacao), inst.classe)
                    
                
                # itera na parte nao sampleada do conj treino
                for inst_test in test:
                    #classifica
                    resposta = getattr(base_learner.classificador, base_learner.rank_method)(base_learner.encoder(inst_test.representacao))
                    
                    rank = util.ranked(resposta) # rank recebe varias tuplas ordenadas ('classe', somaRespostasDosDiscriminadores)
                    # top_score recebe a maior soma dos discriminadores
                    try:
                        top_score = len(rank[0][1])
                    except TypeError:
                        top_score = rank[0][1]
                
                    # coloca na matriz de confusao a resposta e o valor da soma dos discriminadores
                    base_learner.mat_confusao.add(inst_test.classe, rank[0][0], top_score)
                    base_learner.mat_confusao_folds[fold.numero].add(inst_test.classe, rank[0][0], top_score)# essa sera usada pro ensemble com peso por performance
                    
                    
                    #colocar no set corretas ou erradas
                    if inst_test.classe == rank[0][0]:
                        set_corretas.add(inst_test)
                    else:
                        set_erradas.add(inst_test)
                    
                print "Certas: ", len(set_corretas)
                print "Erradas: ", len(set_erradas)
                
                    
        #temporario
        # imprimir as matrizes de confusao de cada classificador
        for base_learner in self.base_learners:
            print base_learner.mat_confusao
            print base_learner.mat_confusao.stats()
            print base_learner.label
        


class Bagging(object):
    
    '''
    Classe responsavel por rodar o algoritmo e armazenar os resultados e metricas
    '''
    
    def __init__(self, dataset, base_learners, ensembles, tam_treino=0.5):
        '''
        receber:
          dataset
          classificadores
          ensembles
          ** talvez as configs de voting
        '''
        self.dataset = dataset
        self.base_learners = base_learners
        self.ensembles = ensembles
        self.tam_treino = tam_treino
        


    def executa_folds(self):
        for fold in self.dataset.folds:
            for base_learner in self.base_learners:
                # samplear
                pesos = fold.retorna_pesos()
                tam_sample = (len(fold.inst_treino) * self.tam_treino)
                sample = np.random.choice(fold.inst_treino, tam_sample, replace=False, p=pesos)
                test = [inst for inst in fold.inst_treino if inst not in sample]
                
                # itera na parte sampleada do conj treino
                for inst in sample:
                    base_learner.classificador.record(base_learner.encoder(inst.representacao), inst.classe)
                    
                
                # itera na parte nao sampleada do conj treino
                for inst_test in test:
                    #classifica
                    resposta = getattr(base_learner.classificador, base_learner.rank_method)(base_learner.encoder(inst_test.representacao))
                    
                    rank = util.ranked(resposta) # rank recebe varias tuplas ordenadas ('classe', somaRespostasDosDiscriminadores)
                    # top_score recebe a maior soma dos discriminadores
                    try:
                        top_score = len(rank[0][1])
                    except TypeError:
                        top_score = rank[0][1]
                
                    # coloca na matriz de confusao a resposta e o valor da soma dos discriminadores
                    base_learner.mat_confusao.add(inst_test.classe, rank[0][0], top_score)
                    base_learner.mat_confusao_folds[fold.numero].add(inst_test.classe, rank[0][0], top_score)# essa sera usada pro ensemble com peso por performance
                    
                    #print "Classe: ", inst_test.classe, "  -  Resposta: ", resposta, "  -  ", ('ACERTOU' if rank[0][0] == inst_test.classe else 'ERROU')
                    
                    #colocar no set corretas ou erradas
                    
        #temporario
        # imprimir as matrizes de confusao de cada classificador
        for base_learner in self.base_learners:
            print base_learner.mat_confusao
            print base_learner.mat_confusao.stats()
            print base_learner.label
        


    