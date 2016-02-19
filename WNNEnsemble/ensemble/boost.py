'''
Created on 19 de fev de 2016

@author: filipi
'''

'''
fazer os metodos para rodar o treino e o test do boost
'''

class AdaBoost(object):
    
    '''
    Classe responsavel por rodar o algoritmo e armazenar os resultados e metricas
    '''
    
    def __init__(self, dataset, base_learners, ensembles):
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
        





    