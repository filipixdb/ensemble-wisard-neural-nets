'''
Created on 18 de fev de 2016

@author: filipi
'''


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
    
    def __init__(self, inst_treino, inst_test):
        self.inst_treino = inst_treino
        self.inst_test = inst_test
        
    def adiciona_instancia(self, instancia, conjunto):
        if conjunto == 'treino':
            self.inst_treino.append(instancia)
        else:
            self.inst_test.append(instancia)



class DataSet(object):
    
    def __init__(self, nome, features, folds):
        self.nome = nome
        self.features = features
        self.folds = folds
    

    def cria_folds(self, data, tam_features, n_folds = 10):
        #aqui data eh um monte de coisas jah lidas
        #tam_features eh uma lista com o tamanho (bits) de cada feature
        #n_folds eh o total de folds a criar
        #as leituras aqui jah estao embaralhadas ou nao
        # criar os folds
        folds = []
        for f in xrange(n_folds):
            folds.append(Fold([], []))
        
        for i, entrada in enumerate(data):
            classe, representacao = entrada
            instancia = Instancia(classe, representacao, tam_features, id=i)
            
            # adicionar nos folds
            for f in xrange(n_folds):
                if (i % n_folds) == f:
                    folds[f].adiciona_instancia(instancia, 'teste')
                else:
                    folds[f].adiciona_instancia(instancia, 'treino')

        return folds
        
        


        
        