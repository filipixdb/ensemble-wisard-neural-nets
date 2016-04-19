'''
fazer os metodos para rodar o treino e o test do boost
'''

import numpy as np
import wann.util as util


class EnsembleAlgorithm(object):
    
    def __init__(self, dataset, base_learners, single_learners, ensembles, tam_treino, mostra_resultados, com_repeticao=True):
        
        self.dataset = dataset
        self.base_learners = base_learners
        self.single_learners = single_learners
        self.ensembles = ensembles
        self.tam_treino = tam_treino
        self.mostra_resultados = mostra_resultados
        self.com_repeticao = com_repeticao

        self.salva_representacoes_instancias()

    def executa_folds(self):
        raise NotImplementedError("Please Implement this method")

    
    def avalia_instancias_treino(self, fold, base_learner, n_learner):
        erro = 0.0
        set_corretas = set()
        responder = getattr(base_learner.classificador, base_learner.rank_method)
        # itera nas instancias de treino pra ver quais devem aumentar ou diminuir o pesos
        for inst_treino in fold.inst_treino:
            #classifica
            #resposta = getattr(base_learner.classificador, base_learner.rank_method)(base_learner.encoder(inst_treino.junta_features(base_learner.selected_features)))
            resposta = responder(inst_treino.dict_representacoes['base'+str(n_learner)])
            rank = util.ranked(resposta) # rank recebe varias tuplas ordenadas ('classe', somaRespostasDosDiscriminadores)

            # guardar instancias corretas, calcular erro
            if inst_treino.classe == rank[0][0]:
                set_corretas.add(inst_treino)
            else:
                erro += inst_treino.peso
                
        return erro, set_corretas



    def avalia_instancias_nao_treinadas(self, instancias_nao_treinadas, base_learner, n_learner):
        erro = 0.0
        erroPorcentagem = 0.0
        erroProporcao = 0.0
        erroProporcaoDiferenca = 0.0
        erroAtivacao = 0.0
        erroPorcentagemAtivacao = 0.0
        erroProporcaoAtivacao = 0.0
        erroProporcaoDiferencaAtivacao = 0.0
        
        totalPesos = 0.0
        totalPesosPorcentagem = 0.0
        totalPesosProporcao = 0.0
        totalPesosProporcaoDiferenca = 0.0
        totalPesosAtivacao = 0.0
        totalPesosPorcentagemAtivacao = 0.0
        totalPesosProporcaoAtivacao = 0.0
        totalPesosProporcaoDiferencaAtivacao = 0.0
        
        erros = {}
        erros['nenhum'] = 0.0
        erros['porcentagem'] = 0.0
        erros['proporcao'] = 0.0
        erros['proporcaoDiferenca'] = 0.0
        erros['ativacao'] = 0.0
        erros['porcentagem+ativacao'] = 0.0
        erros['porporcao+ativacao'] = 0.0
        erros['proporcaoDiferenca+ativacao'] = 0.0
        
        
        certas = 0.0
        erradas = 0.0
        
        responder = getattr(base_learner.classificador, base_learner.rank_method)
        for inst in instancias_nao_treinadas:
            #classifica
            #resposta = getattr(base_learner.classificador, base_learner.rank_method)(base_learner.encoder(inst.junta_features(base_learner.selected_features)))
            resposta = responder(inst.dict_representacoes['base'+str(n_learner)])
            rank = util.ranked(resposta) # rank recebe varias tuplas ordenadas ('classe', somaRespostasDosDiscriminadores)

            confiancas, ativacao = self.calcula_confianca_voto(resposta, base_learner.n_neurons)

            # guardar instancias corretas, calcular erro
            
            #peso_para_o_erro = inst.peso
            peso_para_o_erro = 1.0
            
            if inst.classe == rank[0][0]:
                pass
                certas+=1
            else:
                erro += peso_para_o_erro
                erroPorcentagem += peso_para_o_erro*confiancas['porcentagem']
                erroProporcao += peso_para_o_erro*confiancas['proporcao']
                erroProporcaoDiferenca += peso_para_o_erro*confiancas['proporcaoDiferenca']
                erroAtivacao += peso_para_o_erro*ativacao
                erroPorcentagemAtivacao += peso_para_o_erro*confiancas['porcentagem']*ativacao
                erroProporcaoAtivacao += peso_para_o_erro*confiancas['proporcao']*ativacao
                erroProporcaoDiferencaAtivacao += peso_para_o_erro*confiancas['proporcaoDiferenca']*ativacao
                erradas+=1
            
            totalPesos += peso_para_o_erro
            totalPesosPorcentagem += peso_para_o_erro*confiancas['porcentagem']
            totalPesosProporcao += peso_para_o_erro*confiancas['proporcao']
            totalPesosProporcaoDiferenca += peso_para_o_erro*confiancas['proporcaoDiferenca']
            totalPesosAtivacao += peso_para_o_erro*ativacao
            totalPesosPorcentagemAtivacao += peso_para_o_erro*confiancas['porcentagem']*ativacao
            totalPesosProporcaoAtivacao += peso_para_o_erro*confiancas['proporcao']*ativacao
            totalPesosProporcaoDiferencaAtivacao += peso_para_o_erro*confiancas['proporcaoDiferenca']*ativacao
            
        #print "Acuracia: ", (certas/(certas+erradas)), "    Peso Certas: ", totalPesos-erro, "    Peso Erradas: ", erro, "    Total Pesos: ", totalPesos, "    Erro Percentual: ", erro/totalPesos
        
        erros['nenhum'] = erradas/(certas+erradas)
        erros['porcentagem'] = erroPorcentagem/totalPesosPorcentagem
        erros['proporcao'] = erroProporcao/totalPesosProporcao
        erros['proporcaoDiferenca'] = erroProporcaoDiferenca/totalPesosProporcaoDiferenca
        erros['ativacao'] = erroAtivacao/totalPesosAtivacao
        erros['porcentagem+ativacao'] = erroPorcentagemAtivacao/totalPesosPorcentagemAtivacao
        erros['proporcao+ativacao'] = erroProporcaoAtivacao/totalPesosProporcaoAtivacao
        erros['proporcaoDiferenca+ativacao'] = erroProporcaoDiferencaAtivacao/totalPesosProporcaoDiferencaAtivacao
        
        return erros

    
    
    def atualiza_peso_learner(self, erros, n_learner):
        
        for ens in self.ensembles:
            erro = erros[ens.tipo_erro]
            
            beta = erro/(1.0-erro)
            inverso_beta = 1.0/beta
            log_inverso_beta = np.log(inverso_beta)
            
            ens.guarda_peso(n_learner, log_inverso_beta)
            
        #print " Erro: ", erro, "    Inverso Beta: ", inverso_beta, "    Log Inverso Beta: ", log_inverso_beta



    def exibe_resultados(self):
        print "\n"
        '''
        if len(self.single_learners) > 0:
            print "      Single Learners   -----------------"
            print "\n"
        for single_learner in self.single_learners:
            print "        ", single_learner.label
            #print single_learner.mat_confusao
            print "            ", single_learner.mat_confusao.stats('simples')
        '''
        
        if len(self.single_learners) > 0:
            print "      Single Learners (GERAL)   ------------------"
            print "\n"
        for single_learner in self.single_learners:
            print "        ", single_learner.label
            #print single_learner.mat_confusao
            print "            ", single_learner.mat_confusao_geral.stats('simples', [1, 5])
        
        
        if len(self.base_learners) > 0:
            print "      Base Learners   -----------------"
            print "\n"
        for base_learner in self.base_learners:
            print "        ", base_learner.label
            #print base_learner.mat_confusao
            print "            ", base_learner.mat_confusao.stats('simples', [1, 5])
        
        '''
        if len(self.ensembles) > 0:
            print "      Ensembles   ------------------"
            print "\n"
        for ens in self.ensembles:
            print "        ", ens.label
            #print ens.mat_confusao
            print "            ", ens.mat_confusao.stats('simples')
        
        
        '''
        if len(self.ensembles) > 0:
            print "      Ensembles (GERAL)  ------------------"
            print "\n"
        for ens in self.ensembles:
            print "        ", ens.label
            #print ens.mat_confusao_geral
            #TODO: colocar pra descobrir sozinho os custos
            print "            ", ens.mat_confusao_geral.stats('simples', [1, 5])
        




    def salva_resultados(self):
        with open(self.dataset.nome, 'a') as arq:
            arq.write("\n")
            '''
            if len(self.single_learners) > 0:
                arq.write("\n      Single Learners   -----------------\n")
                arq.write("\n\n")
            for single_learner in self.single_learners:
                arq.write("        "+single_learner.label+'\n')
                #arq.write(single_learner.mat_confusao+'\n')
                arq.write("            "+str(single_learner.mat_confusao.stats('simples'))+'\n')
            '''
            
            if len(self.single_learners) > 0:
                arq.write("\n      Single Learners (GERAL)   ------------------\n")
                arq.write("\n\n")
            for single_learner in self.single_learners:
                arq.write("        "+single_learner.label+'\n')
                #print single_learner.mat_confusao
                arq.write("            "+str(single_learner.mat_confusao_geral.stats('simples', [1, 5]))+'\n')
            
            
            if len(self.base_learners) > 0:
                arq.write("\n      Base Learners   -----------------\n")
                arq.write("\n\n")
            for base_learner in self.base_learners:
                arq.write("        "+base_learner.label+'\n')
                #arq.write(base_learner.mat_confusao+'\n')
                arq.write("            "+str(base_learner.mat_confusao.stats('simples', [1, 5]))+'\n')
            
            '''
            if len(self.ensembles) > 0:
                arq.write("\n      Ensembles   ------------------\n")
                arq.write("\n\n")
            for ens in self.ensembles:
                arq.write("        "+ens.label+'\n')
                #arq.write(ens.mat_confusao+'\n')
                arq.write("            "+str(ens.mat_confusao.stats('simples'))+'\n')
            
            
            '''
            if len(self.ensembles) > 0:
                arq.write("\n      Ensembles (GERAL)  ------------------\n")
                arq.write("\n\n")
            for ens in self.ensembles:
                arq.write("        "+ens.label+'\n')
                #arq.write(ens.mat_confusao_geral+'\n')
                #TODO: colocar pra descobrir sozinho os custos
                arq.write("            "+str(ens.mat_confusao_geral.stats('simples', [1, 5]))+'\n')
        






    def calcula_confianca_voto(self, resposta, n_neuronios):
        
        ativacao = 0.0
        confiancas = {}
        confiancas['porcentagem'] = 0.0 
        confiancas['proporcao'] = 0.0
        confiancas['proporcaoDiferenca'] = 0.0
        
        # calcular a confianca do base learner
        confianca0 = float(resposta['0'])
        confianca1 = float(resposta['1'])
        
        ativados0 = float(resposta['0'])
        ativados1 = float(resposta['1'])
        
        if (confianca0 == 0) or (confianca1 == 0):
            confianca0 += 1.0
            confianca1 += 1.0
        
        somaConfiancas = confianca0+confianca1
#        confianca0 = confianca0/somaConfiancas
#        confianca1 = confianca1/somaConfiancas
        if confianca0 > confianca1:
            confiancas['porcentagem'] = confianca0/somaConfiancas
            confiancas['proporcao'] = confianca0/confianca1
            confiancas['proporcaoDiferenca'] = (confianca0-confianca1)/confianca0
            ativacao = ativados0/n_neuronios
        else:
            confiancas['porcentagem'] = confianca1/somaConfiancas
            confiancas['proporcao'] = confianca1/confianca0
            confiancas['proporcaoDiferenca'] = (confianca1-confianca0)/confianca1
            ativacao = ativados1/n_neuronios
        
        return confiancas, ativacao






    def avalia_base_learner(self, fold, learner, n_learner):
        # avaliar o learner no conjunto de test do fold
        responder = getattr(learner.classificador, learner.rank_method)
        for n_inst, inst_test in enumerate(fold.inst_test):
            #classifica
            #resposta = getattr(learner.classificador, learner.rank_method)(learner.encoder(inst_test.junta_features(learner.selected_features)))
            resposta = responder(inst_test.dict_representacoes['base'+str(n_learner)])
            rank = util.ranked(resposta) # rank recebe varias tuplas ordenadas ('classe', somaRespostasDosDiscriminadores)
            
            #TODO: colocar pra pegar a confianca da forma desejada
            confiancas, ativacao = self.calcula_confianca_voto(resposta, learner.n_neurons)
            
            '''
            # top_score recebe a maior soma dos discriminadores
            try:
                top_score = len(rank[0][1])
            except TypeError:
                top_score = rank[0][1]
            '''
            
            # coloca na matriz de confusao
            learner.mat_confusao.add(inst_test.classe, rank[0][0], 1)
            learner.mat_confusao_folds[fold.numero].add(inst_test.classe, rank[0][0], 1)
            learner.mat_confusao_geral.add(inst_test.classe, rank[0][0], 1)
            

            # guarda os votos do classificador
            #TODO: aqui colocar pra guardar o voto de acordo com o tipo de confianca usado
            for ens in self.ensembles:
                
                if ens.tipo_intensidade == 'nenhum':
                    intensidade = 1.0
                elif ens.tipo_intensidade == 'porcentagem':
                    intensidade = confiancas['porcentagem']
                elif ens.tipo_intensidade == 'proporcao':
                    intensidade = confiancas['proporcao']
                elif ens.tipo_intensidade == 'proporcaoDiferenca':
                    intensidade = confiancas['proporcaoDiferenca']
                elif ens.tipo_intensidade == 'ativacao':
                    intensidade = ativacao
                elif ens.tipo_intensidade == 'porcentagem+ativacao':
                    intensidade = confiancas['porcentagem']*ativacao
                elif ens.tipo_intensidade == 'proporcao+ativacao':
                    intensidade = confiancas['proporcao']*ativacao
                elif ens.tipo_intensidade == 'proporcaoDiferenca+ativacao':
                    intensidade = confiancas['proporcaoDiferenca']*ativacao
                else:
                    print "Erro, tipo_intensidade nao existe -> ", ens.tipo_intensidade
                
                ens.guarda_voto(n_learner, n_inst, rank[0][0], intensidade)
                
                '''
                if ens.com_confiancas == True:
                    ens.guarda_voto(n_learner, n_inst, rank[0][0], confianca*ativados)
                    #print "Resposta: ", resposta, "    Confianca: ", confiancaCorreta, "    Ativados: ", ativadosCorreto, "    Conf*Ativ: ", confiancaCorreta*ativadosCorreto
                else:
                    ens.guarda_voto(n_learner, n_inst, rank[0][0], 1)
                '''



    def avalia_single_learner(self, fold, learner, n_learner):
        # avaliar o learner no conjunto de test do fold
        responder = getattr(learner.classificador, learner.rank_method)
        for n_inst, inst_test in enumerate(fold.inst_test):
            #classifica
            resposta = responder(inst_test.dict_representacoes['single'+str(n_learner)])
            rank = util.ranked(resposta) # rank recebe varias tuplas ordenadas ('classe', somaRespostasDosDiscriminadores)
            
            '''
            # top_score recebe a maior soma dos discriminadores
            try:
                top_score = len(rank[0][1])
            except TypeError:
                top_score = rank[0][1]
            '''
            
            # coloca na matriz de confusao
            learner.mat_confusao.add(inst_test.classe, rank[0][0], 1)
            learner.mat_confusao_folds[fold.numero].add(inst_test.classe, rank[0][0], 1)
            learner.mat_confusao_geral.add(inst_test.classe, rank[0][0], 1)
            


    def avalia_ensembles(self, fold):

        for ens in self.ensembles:
            ens.inicia_agregador(self.dataset.n_classes)
            ens.predict()
            
            # avaliar os ensembles
            for n_inst, inst_test in enumerate(fold.inst_test):
                y1 = inst_test.classe
                y2 = str(ens.combined_votes[n_inst])
                ens.mat_confusao_folds[fold.numero].add(y1, y2, 1)
                ens.mat_confusao.add(y1, y2, 1)
                ens.mat_confusao_geral.add(y1, y2, 1)




    def treina_base_learner(self, fold, base_learner, n_learner, com_repeticao):
        # samplear
        tam_sample = (len(fold.inst_treino) * self.tam_treino)
        pesos = fold.retorna_pesos()
        
        sample = np.random.choice(fold.inst_treino, tam_sample, replace=com_repeticao, p=pesos)
        nao_treinadas = []
        
        gravar = base_learner.classificador.record
        
        '''
        for inst in fold.inst_treino:
            if inst in sample:
                gravar(inst.dict_representacoes['base'+str(n_learner)], inst.classe)
            else:
                nao_treinadas.append(inst)
        '''
        
        for inst in sample:
            gravar(inst.dict_representacoes['base'+str(n_learner)], inst.classe)
        
        nao_treinadas = [inst for inst in set(fold.inst_treino) - set(sample)]
        
        
        '''
        # itera na parte sampleada do conj treino
        for inst in sample:
            base_learner.classificador.record(base_learner.encoder(inst.junta_features(base_learner.selected_features)), inst.classe)
        '''
        
        return nao_treinadas

    def treina_single_learner(self, fold, single_learner, n_learner):
        # itera em todo conj treino
        for inst_treino in fold.inst_treino:
            single_learner.classificador.record(inst_treino.dict_representacoes['single'+str(n_learner)], inst_treino.classe)

    def reseta_classificadores(self, learners):
        # itera na lista de learners para reseta-los a cada fold
        for learner in learners:
            learner.reseta_classificador()


    def salva_representacoes_instancias(self):
        for n_learner, learner in enumerate(self.single_learners):
            encode = learner.encoder
            features = learner.selected_features
            for fold in self.dataset.folds:
                for inst in fold.inst_test + fold.inst_treino:
                    representacao = encode(inst.junta_features(features))
                    inst.salva_representacao('single'+str(n_learner), representacao)
        for n_learner, learner in enumerate(self.base_learners):
            encode = learner.encoder
            features = learner.selected_features
            for fold in self.dataset.folds:
                for inst in fold.inst_test + fold.inst_treino:
                    representacao = encode(inst.junta_features(features))
                    inst.salva_representacao('base'+str(n_learner), representacao)
                    

class AdaBoost(EnsembleAlgorithm):
    
    '''
    Classe responsavel por rodar o algoritmo e armazenar os resultados e metricas
    '''
    
    def __init__(self, dataset, base_learners, single_learners, ensembles, tam_treino, mostra_resultados, com_repeticao=True):
        
        super(AdaBoost, self).__init__(dataset, base_learners, single_learners, ensembles, tam_treino, mostra_resultados, com_repeticao)


    def executa_folds(self):
        for fold in self.dataset.folds:
            
            self.dataset.reseta_pesos()
            
            # reseta os classificadores a cada novo fold
            self.reseta_classificadores(self.base_learners)
            self.reseta_classificadores(self.single_learners)

            
            # treinar e avaliar os single learners
            for n_single_learner, single_learner in enumerate(self.single_learners):
                self.treina_single_learner(fold, single_learner, n_single_learner)
                self.avalia_single_learner(fold, single_learner, n_single_learner)
            
            
            for ens in self.ensembles:
                ens.inicia_votos_e_pesos(len(self.base_learners), len(fold.inst_test))
            
            for n_learner, base_learner in enumerate(self.base_learners):
                
                instancias_nao_treinadas = self.treina_base_learner(fold, base_learner, n_learner, self.com_repeticao)

                erroConjTreino, set_corretas = self.avalia_instancias_treino(fold, base_learner, n_learner)
                
                
                #TODO: remover esta gambiarra
                #instancias_nao_treinadas = fold.inst_treino

                erros = self.avalia_instancias_nao_treinadas(instancias_nao_treinadas, base_learner, n_learner)
                self.atualiza_peso_learner(erros, n_learner)
                
                # nao pode ser antes de avaliar as instancias nao treinadas
                self.atualiza_pesos_instancias_treino(fold, set_corretas, erroConjTreino)
                
                                
                self.avalia_base_learner(fold, base_learner, n_learner)
                
                
            self.avalia_ensembles(fold)            
        
        if (self.mostra_resultados):
            self.exibe_resultados()
            self.salva_resultados()

        
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
    
    def __init__(self, dataset, base_learners, single_learners, ensembles, tam_treino, mostra_resultados, com_repeticao=True):
        super(Bagging, self).__init__(dataset, base_learners, single_learners, ensembles, tam_treino, mostra_resultados, com_repeticao)
    

    def executa_folds(self):
        for fold in self.dataset.folds:
            
            self.dataset.reseta_pesos()
            
            # reseta os classificadores a cada novo fold
            self.reseta_classificadores(self.base_learners)
            self.reseta_classificadores(self.single_learners)
            
            # treinar e avaliar os single learners
            for n_single_learner, single_learner in enumerate(self.single_learners):
                self.treina_single_learner(fold, single_learner, n_single_learner)
                self.avalia_single_learner(fold, single_learner, n_single_learner)
                        
            for ens in self.ensembles:
                ens.inicia_votos_e_pesos(len(self.base_learners), len(fold.inst_test))
                
            
            for n_learner, base_learner in enumerate(self.base_learners):
                
                instancias_nao_treinadas = self.treina_base_learner(fold, base_learner, n_learner, self.com_repeticao)

                # calcular o erro do classificador no conj de treino
                #erroConjTreino, _ = self.avalia_instancias_treino(fold, base_learner)

                #TODO: remover esta gambiarra
                #instancias_nao_treinadas = fold.inst_treino
                
                # calcular o erro nas instancias nao treinadas
                erros = self.avalia_instancias_nao_treinadas(instancias_nao_treinadas, base_learner, n_learner)
                
                self.atualiza_peso_learner(erros, n_learner)
                
                
                self.avalia_base_learner(fold, base_learner, n_learner)
                
                
            self.avalia_ensembles(fold)            
        
        if (self.mostra_resultados):
            self.exibe_resultados()
            self.salva_resultados()
