'''
Created on 18 de fev de 2016

@author: filipi
'''
from ensemble.partition import cria_folds


'''
Aqui fazer as classes para
  dataset
    nome
    features
    folds
  fold
    instancias treino
    instancias teste
  instancia
    id
    classe
    representacao
    features
    peso
'''


class Instancia(object):
    
    def __init__(self, classe, representacao, features, peso=1, identificador=None):
        #fazer auto gerar identificador unico
        self.classe = classe
        self.representacao = representacao
        self.features = features
        self.peso = peso
        
        if identificador is not None:
            self.identificador = identificador
        else:
            # autogerar ids unicos
            #TODO: criar metodo estatico que gera os ids
            pass


    

class Fold(object):
    
    def __init__(self, inst_treino, inst_test, numero):
        self.inst_treino = inst_treino
        self.inst_test = inst_test
        self.numero = numero
        
    def adiciona_instancia(self, instancia, conjunto):
        if conjunto == 'treino':
            self.inst_treino.append(instancia)
        else:
            self.inst_test.append(instancia)
            
    def adiciona_pesos(self, pesos):
        for i, peso in enumerate(pesos):
            self.inst_treino[i].peso = peso
    
    def retorna_pesos(self):
        pesos = []
        for inst in self.inst_treino:
            pesos.append(inst.peso)



class DataSet(object):
    
    def __init__(self, data, tam_features, n_folds, nome=None):
        self.nome = nome
        self.features = tam_features
        self.folds = n_folds
        
        self.cria_folds(data, tam_features, n_folds)
    

    def cria_folds(self, data, tam_features, n_folds = 10):
        #aqui data eh um monte de coisas jah lidas
        #tam_features eh uma lista com o tamanho (bits) de cada feature
        #n_folds eh o total de folds a criar
        #as leituras aqui jah estao embaralhadas ou nao
        # criar os folds
        self.folds = []
        for f in xrange(n_folds):
            self.folds.append(Fold([], [], f))
        
        
        for i, entrada in enumerate(data):
            numero, classe, representacao = entrada
            instancia = Instancia(classe, representacao, tam_features, identificador=numero)
            
            # adicionar nos folds
            for f in xrange(n_folds):
                if (i % n_folds) == f:
                    self.folds[f].adiciona_instancia(instancia, 'teste')
                else:
                    self.folds[f].adiciona_instancia(instancia, 'treino')
        
        #inicializa pesos iguais
        for fold in self.folds:
            peso = 1.0 / len(fold.inst_treino)
            fold.adiciona_pesos([peso] * len(fold.inst_treino))

        
        


        
        