'''
fazer os metodos para rodar o treino e o test do boost
'''

import numpy as np
import wann.util as util


class EnsembleAlgorithm(object):
    
    def __init__(self, dataset, base_learners, single_learners, ensembles, tam_treino, com_repeticao=True):
        
        self.dataset = dataset
        self.base_learners = base_learners
        self.single_learners = single_learners
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
            resposta = getattr(base_learner.classificador, base_learner.rank_method)(base_learner.encoder(inst_treino.junta_features(base_learner.selected_features)))
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

    '''
    def atualiza_peso_learner_erro_relativo(self, erro, n_learner):
        beta = erro/(1.0-erro)
        log_erro = np.log(1/beta)
        
        #print "  Log Erro: ", log_erro
        
        for ens in self.ensembles:
            ens.guarda_peso(n_learner, log_erro)
    '''

    '''
    def ajusta_pesos_ensemble(self):
        for ens in self.ensembles:
            pesoTotal = 0.0
            for n_learner, learner in enumerate(self.base_learners):
                pesoTotal += ens.pesos_learners[n_learner]
            
            media = pesoTotal/len(self.base_learners)
            
            for n_learner, learner in enumerate(self.base_learners):
                peso = ens.pesos_learners[n_learner] - media
                #peso = ens.pesos_learners[n_learner]/pesoTotal
                ens.guarda_peso(n_learner, peso)
            
            print "  Pesos Learners: "
            print ens.pesos_learners
    '''

    def exibe_resultados(self):
        '''
        if len(self.single_learners) > 0:
            print "  Single Learners --------------------"
        for single_learner in self.single_learners:
            print "    ", single_learner.label
            #print single_learner.mat_confusao
            print "        ", single_learner.mat_confusao.stats('simples')
        '''
        if len(self.single_learners) > 0:
            print "  Single Learners (GERAL)--------------------"
        for single_learner in self.single_learners:
            print "    ", single_learner.label
            #print single_learner.mat_confusao
            print "        ", single_learner.mat_confusao_geral.stats('simples')
        
        if len(self.base_learners) > 0:
            print "  Base Learners ----------------------"
        for base_learner in self.base_learners:
            print "    ", base_learner.label
            #print base_learner.mat_confusao
            print "        ", base_learner.mat_confusao.stats('simples')
            
        '''
        if len(self.ensembles) > 0:
            print "  Ensembles --------------------------"
        for ens in self.ensembles:
            print "    ", ens.label
            #print ens.mat_confusao
            print "        ", ens.mat_confusao.stats('simples')
        
        
        '''
        if len(self.ensembles) > 0:
            print "  Ensembles (GERAL)--------------------------"
        for ens in self.ensembles:
            print "    ", ens.label
            #print ens.mat_confusao_geral
            print "        ", ens.mat_confusao_geral.stats('simples')
        

    def avalia_base_learner(self, fold, learner, n_learner):
        erro = 0.0
        erroTotal = 0.0
        # avaliar o learner no conjunto de test do fold
        for n_inst, inst_test in enumerate(fold.inst_test):
            #classifica
            resposta = getattr(learner.classificador, learner.rank_method)(learner.encoder(inst_test.junta_features(learner.selected_features)))
            rank = util.ranked(resposta) # rank recebe varias tuplas ordenadas ('classe', somaRespostasDosDiscriminadores)
            
            
            # calcular a confianca do base learner
            confianca0 = resposta['0']*1.0
            confianca1 = resposta['1']*1.0
            
            if (confianca0 == 0) and (confianca1 == 0):
                confianca0 = 0.5
                confianca1 = 0.5
            elif (confianca0 == 0) or (confianca1 == 0):
                if (confianca0 == 0):
                    confianca0 = 0.1
                    confianca1 = 0.9
                else:
                    confianca0 = 0.9
                    confianca1 = 0.1
            
            somaConfiancas = confianca0+confianca1
            confianca0 = confianca0/somaConfiancas
            confianca1 = confianca1/somaConfiancas
            if confianca0 > confianca1:
                confiancaCorreta = confianca0
            else:
                confiancaCorreta = confianca1
            
            
            # top_score recebe a maior soma dos discriminadores
            try:
                top_score = len(rank[0][1])
            except TypeError:
                top_score = rank[0][1]
        
            # coloca na matriz de confusao
            learner.mat_confusao.add(inst_test.classe, rank[0][0], top_score)
            learner.mat_confusao_folds[fold.numero].add(inst_test.classe, rank[0][0], top_score)
            learner.mat_confusao_geral.add(inst_test.classe, rank[0][0], top_score)
            

            # guarda os votos do classificador
            for ens in self.ensembles:
                if ens.com_confiancas == True:
                    ens.guarda_voto(n_learner, n_inst, rank[0][0], confiancaCorreta)
                else:
                    ens.guarda_voto(n_learner, n_inst, rank[0][0], 1)
            
            # calcular total possivel para o erro
            erroTotal += inst_test.peso
            # calcular erro
            if inst_test.classe != rank[0][0]:
                erro += inst_test.peso
            
            
        return erro/erroTotal


    def avalia_single_learner(self, fold, learner, n_learner):
        # avaliar o learner no conjunto de test do fold
        for n_inst, inst_test in enumerate(fold.inst_test):
            #classifica
            resposta = getattr(learner.classificador, learner.rank_method)(learner.encoder(inst_test.junta_features(learner.selected_features)))
            rank = util.ranked(resposta) # rank recebe varias tuplas ordenadas ('classe', somaRespostasDosDiscriminadores)
            
            # top_score recebe a maior soma dos discriminadores
            try:
                top_score = len(rank[0][1])
            except TypeError:
                top_score = rank[0][1]
        
            # coloca na matriz de confusao
            learner.mat_confusao.add(inst_test.classe, rank[0][0], top_score)
            learner.mat_confusao_folds[fold.numero].add(inst_test.classe, rank[0][0], top_score)
            learner.mat_confusao_geral.add(inst_test.classe, rank[0][0], top_score)
            


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
                ens.mat_confusao_geral.add(y1, y2, 0)


    def treina_base_learner(self, fold, base_learner, com_repeticao):
        # samplear
        tam_sample = (len(fold.inst_treino) * self.tam_treino)
        pesos = fold.retorna_pesos()
        sample = np.random.choice(fold.inst_treino, tam_sample, replace=com_repeticao, p=pesos)
        
        # itera na parte sampleada do conj treino
        for inst in sample:
            base_learner.classificador.record(base_learner.encoder(inst.junta_features(base_learner.selected_features)), inst.classe)


    def treina_single_learner(self, fold, single_learner):
        # itera em todo conj treino
        for inst_treino in fold.inst_treino:
            single_learner.classificador.record(single_learner.encoder(inst_treino.junta_features(single_learner.selected_features)), inst_treino.classe)

    def reseta_classificadores(self, learners):
        # itera na lista de learners para reseta-los a cada fold
        for learner in learners:
            learner.reseta_classificador()


class AdaBoost(EnsembleAlgorithm):
    
    '''
    Classe responsavel por rodar o algoritmo e armazenar os resultados e metricas
    '''
    
    def __init__(self, dataset, base_learners, single_learners, ensembles, tam_treino, com_repeticao=True):
        
        super(AdaBoost, self).__init__(dataset, base_learners, single_learners, ensembles, tam_treino, com_repeticao)


    def executa_folds(self):
        for fold in self.dataset.folds:
            
            self.dataset.reseta_pesos()
            
            # reseta os classificadores a cada novo fold
            self.reseta_classificadores(self.base_learners)
            self.reseta_classificadores(self.single_learners)

            
            # treinar e avaliar os single learners
            for n_single_learner, single_learner in enumerate(self.single_learners):
                self.treina_single_learner(fold, single_learner)
                self.avalia_single_learner(fold, single_learner, n_single_learner)
            
            
            for ens in self.ensembles:
                ens.inicia_votos_e_pesos(len(self.base_learners), len(fold.inst_test))
            
            for n_learner, base_learner in enumerate(self.base_learners):
                
                self.treina_base_learner(fold, base_learner, self.com_repeticao)

                erro, set_corretas = self.avalia_instancias_treino(fold, base_learner)
                
                # usando da forma abaixo consideraria o erro no conj de treino para calcular o peso do classificador
                # usando da outra forma, considera o erro no conj de test
                
                self.atualiza_peso_learner(erro, n_learner)
                
                self.atualiza_pesos_instancias_treino(fold, set_corretas, erro)
                
                _ = self.avalia_base_learner(fold, base_learner, n_learner)
                
                '''
                self.atualiza_pesos_instancias_treino(fold, set_corretas, erro)
                
                erro = self.avalia_base_learner(fold, base_learner, n_learner)
                
                self.atualiza_peso_learner(erro, n_learner)
                '''
                
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
    
    def __init__(self, dataset, base_learners, single_learners, ensembles, tam_treino, com_repeticao=True):
        super(Bagging, self).__init__(dataset, base_learners, single_learners, ensembles, tam_treino, com_repeticao)
    

    def executa_folds(self):
        for fold in self.dataset.folds:
            
            self.dataset.reseta_pesos()
            
            # reseta os classificadores a cada novo fold
            self.reseta_classificadores(self.base_learners)
            self.reseta_classificadores(self.single_learners)
            
            # treinar e avaliar os single learners
            for n_single_learner, single_learner in enumerate(self.single_learners):
                self.treina_single_learner(fold, single_learner)
                self.avalia_single_learner(fold, single_learner, n_single_learner)
                        
            for ens in self.ensembles:
                ens.inicia_votos_e_pesos(len(self.base_learners), len(fold.inst_test))
                
            
            for n_learner, base_learner in enumerate(self.base_learners):
                
                self.treina_base_learner(fold, base_learner, self.com_repeticao)

                erro = self.avalia_base_learner(fold, base_learner, n_learner)
                # Na pratica nunca cai aqui, mas isso existe pq se o conj de teste ficar pequeno, pode nao ter erro
                # Se nao der erro, quebra por divisao por zero
                if erro == 0.0:
                    erro += 0.00001
                    print "Erro zero"
                
                self.atualiza_peso_learner(erro, n_learner)
            
                
            self.avalia_ensembles(fold)            
        
        self.exibe_resultados()

