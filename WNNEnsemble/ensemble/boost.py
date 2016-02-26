'''
@author: filipi
'''

'''
fazer os metodos para rodar o treino e o test do boost
'''

import numpy as np
import wann.util as util


class EnsembleAlgorithm(object):
    
    def __init__(self, dataset, base_learners, ensembles, tam_treino, com_repeticao=True):
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
        self.com_repeticao = com_repeticao


    def executa_folds(self):
        raise NotImplementedError("Please Implement this method")

    
    def avalia_instancias_treino(self, fold, base_learner):
        erro = 0.0
        set_corretas = set()
        # itera nas instancias de treino pra ver quais devem aumentar ou diminuir o pesos
        for inst_treino in fold.inst_treino:
            #classifica
            resposta = getattr(base_learner.classificador, base_learner.rank_method)(base_learner.encoder(inst_treino.representacao))
            rank = util.ranked(resposta) # rank recebe varias tuplas ordenadas ('classe', somaRespostasDosDiscriminadores)

            # guardar instancias corretas, calcular erro
            if inst_treino.classe == rank[0][0]:
                set_corretas.add(inst_treino)
            else:
                erro += inst_treino.peso
                
        return erro, set_corretas
    
    
    def atualiza_peso_learner(self, erro, n_learner):
        beta = erro/(1.0-erro)
        log_erro = np.log(1/beta)
        
        for ens in self.ensembles:
            ens.guarda_peso(n_learner, log_erro)


    def exibe_resultados(self):
        for base_learner in self.base_learners:
            print base_learner.mat_confusao
            print base_learner.mat_confusao.stats()
            print base_learner.label
            
        for ens in self.ensembles:
            print ens.mat_confusao
            print ens.mat_confusao.stats()
            print ens.label


    def avalia_learner(self, fold, base_learner, n_learner):
        erro = 0.0
        # avaliar o learner no conjunto de test do fold
        for n_inst, inst_test in enumerate(fold.inst_test):
            #classifica
            resposta = getattr(base_learner.classificador, base_learner.rank_method)(base_learner.encoder(inst_test.representacao))
            rank = util.ranked(resposta) # rank recebe varias tuplas ordenadas ('classe', somaRespostasDosDiscriminadores)
            
            # top_score recebe a maior soma dos discriminadores
            try:
                top_score = len(rank[0][1])
            except TypeError:
                top_score = rank[0][1]
        
            # coloca na matriz de confusao
            base_learner.mat_confusao.add(inst_test.classe, rank[0][0], top_score)
            base_learner.mat_confusao_folds[fold.numero].add(inst_test.classe, rank[0][0], top_score)
            
            # guarda os votos do classificador
            for ens in self.ensembles:
                ens.guarda_voto(n_learner, n_inst, rank[0][0])
            
            # calcular erro
            if inst_test.classe != rank[0][0]:
                erro += inst_test.peso
            
            
        return erro


    def avalia_ensembles(self, fold):
        # inicia combinadores
        for ens in self.ensembles:
            ens.inicia_agregador(self.dataset.n_classes)
            ens.predict()
            
        # avaliar os ensembles
        for n_inst, inst_test in enumerate(fold.inst_test):
            for ens in self.ensembles:
                y1 = inst_test.classe
                y2 = str(ens.combined_votes[n_inst])
                ens.mat_confusao_folds[fold.numero].add(y1, y2, 0)
                ens.mat_confusao.add(y1, y2, 0)


    def treina_learner(self, fold, base_learner, com_repeticao):
        # samplear
        tam_sample = (len(fold.inst_treino) * self.tam_treino)
        pesos = fold.retorna_pesos()
        sample = np.random.choice(fold.inst_treino, tam_sample, replace=com_repeticao, p=pesos)
        
        # itera na parte sampleada do conj treino
        for inst in sample:
            base_learner.classificador.record(base_learner.encoder(inst.representacao), inst.classe)



class AdaBoost(EnsembleAlgorithm):
    
    '''
    Classe responsavel por rodar o algoritmo e armazenar os resultados e metricas
    '''
    
    def __init__(self, dataset, base_learners, ensembles, tam_treino, com_repeticao=True):
        
        super(AdaBoost, self).__init__(dataset, base_learners, ensembles, tam_treino, com_repeticao)


    def executa_folds(self):
        for fold in self.dataset.folds:
            
            self.dataset.reseta_pesos()
            
            for ens in self.ensembles:
                ens.inicia_votos_e_pesos(len(self.base_learners), len(fold.inst_test))
            
            for n_learner, base_learner in enumerate(self.base_learners):
                
                self.treina_learner(fold, base_learner, self.com_repeticao)

                erro, set_corretas = self.avalia_instancias_treino(fold, base_learner)
                
                self.atualiza_peso_learner(erro, n_learner)
                
                self.atualiza_pesos_instancias_treino(fold, set_corretas, erro)
                
                _ = self.avalia_learner(fold, base_learner, n_learner)
                
            self.avalia_ensembles(fold)            
        
        self.exibe_resultados()

        
    def atualiza_pesos_instancias_treino(self, fold, set_corretas, erro):
        beta = erro/(1.0-erro)
        
        # atualizar os pesos
        total_pesos_temp = 0.0
        for inst_treino in fold.inst_treino:
            if inst_treino in set_corretas:
                inst_treino.peso = (inst_treino.peso * beta)
            total_pesos_temp += inst_treino.peso
            
        # normalizar os novos pesos das instancias
        for inst_treino in fold.inst_treino:
            inst_treino.peso = (inst_treino.peso / total_pesos_temp)



class Bagging(EnsembleAlgorithm):
    
    '''
    Classe responsavel por rodar o algoritmo e armazenar os resultados e metricas
    '''
    
    def __init__(self, dataset, base_learners, ensembles, tam_treino, com_repeticao=True):
        super(Bagging, self).__init__(dataset, base_learners, ensembles, tam_treino, com_repeticao)
    

    def executa_folds(self):
        for fold in self.dataset.folds:
            
            self.dataset.reseta_pesos()
            
            for ens in self.ensembles:
                ens.inicia_votos_e_pesos(len(self.base_learners), len(fold.inst_test))
            
            for n_learner, base_learner in enumerate(self.base_learners):
                
                self.treina_learner(fold, base_learner, self.com_repeticao)

                erro = self.avalia_learner(fold, base_learner, n_learner)
                if erro == 0.0:
                    erro += 0.00001
                    print "Erro zero"
                
                self.atualiza_peso_learner(erro, n_learner)
                
            self.avalia_ensembles(fold)            
        
        self.exibe_resultados()

