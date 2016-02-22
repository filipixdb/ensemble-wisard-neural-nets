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
            pesos_learners = []
            for base_learner in self.base_learners:
                # samplear
                pesos = fold.retorna_pesos()
                tam_sample = (len(fold.inst_treino) * self.tam_treino)
                sample = np.random.choice(fold.inst_treino, tam_sample, replace=False, p=pesos)
                #test = [inst for inst in fold.inst_treino if inst not in sample]
                
                set_corretas = set()
                set_erradas = set()
                erro = 0.0
                peso_corretos = 0.0
                
                # itera na parte sampleada do conj treino
                for inst in sample:
                    base_learner.classificador.record(base_learner.encoder(inst.representacao), inst.classe)
                    
                
                # itera na parte nao sampleada ou entao na parte toda de treino
                # olhar isso com mais calma
                #for inst_test in test:
                for inst_treino in fold.inst_treino:
                    #classifica
                    resposta = getattr(base_learner.classificador, base_learner.rank_method)(base_learner.encoder(inst_treino.representacao))
                    
                    rank = util.ranked(resposta) # rank recebe varias tuplas ordenadas ('classe', somaRespostasDosDiscriminadores)
                    # top_score recebe a maior soma dos discriminadores
                    try:
                        top_score = len(rank[0][1])
                    except TypeError:
                        top_score = rank[0][1]
                
                    #colocar no set corretas ou erradas
                    if inst_treino.classe == rank[0][0]:
                        set_corretas.add(inst_treino)
                        peso_corretos += inst_treino.peso
                    else:
                        set_erradas.add(inst_treino)
                        erro += inst_treino.peso
                pesos_learners.append(np.log((1 - erro)/erro) / 2)
                print np.log((1 - erro)/erro) / 2
                
                
                # atualizar os pesos
                beta = erro/(1.0-erro)
                total_pesos_temp = 0.0
                
                for inst_treino in fold.inst_treino:
                    if inst_treino in set_corretas:
                        inst_treino.peso = (inst_treino.peso * beta)
                    else:
                        inst_treino.peso = (inst_treino.peso)
                    total_pesos_temp += inst_treino.peso
                
                for inst_treino in fold.inst_treino:
                    inst_treino.peso = (inst_treino.peso / total_pesos_temp)
                
                
                
                # avaliar o learner no conjunto de test do fold
                for inst_test in fold.inst_test:
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
        


    