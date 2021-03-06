'''
Estruturas de um dataset
  folds
  instancias
  infos das features
  etc
'''
from copy import copy

class Instancia(object):
    
    def __init__(self, classe, representacao, tam_features, peso=1, identificador=None):
        #fazer auto gerar identificador unico
        self.classe = classe
        self.representacao = representacao
        self.dict_representacoes = {}
        self.tam_features = tam_features
        self.peso = peso
        
        if identificador is not None:
            self.identificador = identificador
        else:
            # autogerar ids unicos
            #TODO: criar metodo estatico que gera os ids
            pass
        
        self.features = self.separa_features()

    def separa_features(self):
        features = []
        k = 0
        for tam in self.tam_features:
            features.append(self.representacao[k:(k+tam)])
            k += tam
        return features
    
    def junta_features(self, indices):
        
        return ''.join([self.features[i] for i in indices])
        '''
        saida = ""
        for i in indices:
            saida += self.features[int(i)]
        return saida
        '''
    
    def salva_representacao(self, key, representacao):
        self.dict_representacoes[key] = representacao

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
        return pesos



class DataSet(object):
    
    def __init__(self, data, tam_features, n_folds, n_classes, nome=None):
        self.nome = nome
        self.tam_features = tam_features
        self.folds = n_folds
        self.n_classes = n_classes
        
        self.cria_folds(data, tam_features, n_folds)
    

    def cria_folds(self, data, tam_features, n_folds = 10):
        # criar os folds
        '''
        self.folds = []
        for f in xrange(n_folds):
            self.folds.append(Fold([], [], f))
        '''    
        self.folds = [Fold([], [], f) for f in xrange(n_folds)]

        instancias = [Instancia(classe, representacao, tam_features, identificador=numero) for (numero, classe, representacao) in data]
        
        for f in xrange(n_folds):
            self.folds[f].inst_treino = [copy(inst) for i,inst in enumerate(instancias) if (i % n_folds) != f]
            self.folds[f].inst_test = [copy(inst) for i,inst in enumerate(instancias) if (i % n_folds) == f]
        
        self.reseta_pesos()

        
    def reseta_pesos(self):
        for fold in self.folds:
            peso = 1.0 / len(fold.inst_treino)
            fold.adiciona_pesos([peso] * len(fold.inst_treino))
        