'''
Created on 19 de fev de 2016

@author: filipi
'''
from data_process.encoding import BitStringEncoder

'''
fazer os metodos para rodar o treino e o test do boost
'''

import numpy as np

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
                
                #for instancia no conjunto de treino
                for inst in sample:
                    base_learner.classificador.record(base_learner.encoder(inst.representacao), inst.classe)
                    
                    print "record ", base_learner.encoder(inst.representacao),"  -  ", inst.classe
        


    