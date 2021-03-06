'''
  Base-learners
  Singles-learners
  Ensembles
  

'''

import time

from _functools import partial
import wann.vgram as wann_vgram
import wann.wisard.classifiers as wann_wis_clf
import wann.wisard.discriminators as wann_wis_dsc
import wann.util as util
import wann.crossval as wann_xval
import ensemble.composition as compo
from data_process.encoding import BitStringEncoder

def cria_classificadores(selected_features, nmbr_neurons_list, tipos_classi=3):
    
    lottery_option = partial(wann_wis_clf.WiSARD,
                             discriminator=wann_wis_dsc.LotteryDiscriminator)
    vgram_option = partial(wann_wis_clf.WiSARD,
                           discriminator=wann_vgram.VGWiSARDDiscriminator)

    classificadores = []
    labels = {}
    chaves_classificadores = []
    
    for z, conj_feat in enumerate(selected_features):
        # criar a chave que identifica unicamente cada classificador
        chave = str(z)
        chaves_classificadores.append(chave)
        # atencao ao mapear o numero de neuronios
        idx_nmr_neuro = (z % len(nmbr_neurons_list))
        neuronios = nmbr_neurons_list[idx_nmr_neuro]
        
        
        tipo_classi = z % tipos_classi
        if tipo_classi == 0:
            classificadores.append((wann_wis_clf.WiSARD, 'WiSARD - neuronios: '+`neuronios`, 'counts', neuronios, chave))# o i eh o numero de neuronios
            labels[chave] = ('WiSARD - neuronios: '+`neuronios`)
        elif tipos_classi == 1:
            classificadores.append((lottery_option, 'WiSARD + LotteryDisc - neuronios: '+`neuronios`, 'answers', neuronios, chave))
            labels[chave] = ('WiSARD + LotteryDisc - neuronios: '+`neuronios`)
        elif tipo_classi == 2:
            classificadores.append((vgram_option, 'VGWiSARDDisc - neuronios: '+`neuronios`, 'answers', neuronios, chave))
            labels[chave] = ('VGWiSARDDisc - neuronios: '+`neuronios`)
        elif tipo_classi == 4:
            pass #MCD

    return classificadores, chaves_classificadores, labels



def inicializa_classificadores(classificadores, fold_data_dict, bagging=None, percent=None):
    confusion_mat = {}
    times = {}
    cv_gen = {}

    # Itera em todas as opcoes de classificadores
    for classifier, label, rank_method, nmbr_neurons, chave in classificadores:
        confusion_mat[chave] = util.ConfusionMatrix()
        time_alfa = time.time()
        # cv_gen recebe um monte de (classificador, (observacao, classificacao, classe_real))
        # aqui ele ja manda a wisard treinar as instancias de treino de cada fold
        cv_gen[chave] = wann_xval.cross_validate(fold_data_dict[chave], classifier, rank_method, bagging, percent)
        times[chave] = (time.time() - time_alfa)
    return confusion_mat, times, cv_gen



def inicializa_matrizes_de_confusao(classificador):
    confusion_mat = {}
    # Itera em todas as opcoes de classificadores
    for classifier, label, rank_method, nmbr_neurons, chave in classificador:
        confusion_mat[chave] = util.ConfusionMatrix()
    return confusion_mat



class BaseLearner(object):
    '''
    Classe que guarda o classifier, seus parametros e matriz de confusao
    '''
    static_encoder = None
    
    def __init__(self, discriminador, n_neurons, bits_endereco, rank_method, n_folds, confusion_matrix_geral, classificador='wisard', mapping_igual = False, selected_features = None):
        '''
        classifier
        label
        rank_method
        n_neurons
        mat_confusao
        mat_confusao_folds
        selected tam_features
        '''
        self.param_classificador = classificador
        self.param_discriminador = discriminador
        self.classificador = cria_classificador(classificador, discriminador)
        
        self.n_neurons = n_neurons
        self.bits_endereco = bits_endereco
        self.rank_method = rank_method
        self.selected_features = selected_features
        
        self.mat_confusao = util.ConfusionMatrix()
        self.mat_confusao_geral = confusion_matrix_geral
        self.mat_confusao_folds = []
        for _ in xrange(n_folds):
            self.mat_confusao_folds.append(util.ConfusionMatrix())
            
        self.label = discriminador+" - neuronios= "+str(n_neurons)+ " - endereco= "+str(bits_endereco)+" - rank= "+rank_method+" - features= "+str(selected_features)

        self.mapping_igual = mapping_igual#necessario se quiser resetar o mapping estatico a cada fold
        if mapping_igual:
            if BaseLearner.static_encoder == None:
                BaseLearner.static_encoder = BitStringEncoder(self.n_neurons)
            self.encoder = BaseLearner.static_encoder
        else:
            self.encoder = BitStringEncoder(self.n_neurons)
        

    def reseta_classificador(self):
        '''
        se quiser resetar o mapping estatico a cada fold
        if self.mapping_igual:
            if BaseLearner.static_encoder == self.encoder:
                BaseLearner.static_encoder = BitStringEncoder(self.n_neurons)
                self.encoder = BaseLearner.static_encoder
            else:
                self.encoder = BaseLearner.static_encoder
        '''    
        if self.param_discriminador == 'wisard':
            disc = wann_wis_dsc.Discriminator
        if self.param_discriminador == 'lottery':
            disc = wann_wis_dsc.LotteryDiscriminator
        if self.param_discriminador == 'vgwisard':
            disc = wann_vgram.VGWiSARDDiscriminator
        
        self.classificador = wann_wis_clf.WiSARD(disc)

    
def cria_classificador(classificador, discriminador):
    
    if discriminador == 'wisard':
        disc = wann_wis_dsc.Discriminator
    if discriminador == 'lottery':
        disc = wann_wis_dsc.LotteryDiscriminator
    if discriminador == 'vgwisard':
        disc = wann_vgram.VGWiSARDDiscriminator
    
    return wann_wis_clf.WiSARD(disc)
    
def cria_learners(configs_learners, n_folds, tam_features, mapping_igual=False):
    
    '''
    recebe uma lista de tuplas com as configs dos base learners
    retorna uma lista com os base learners
    configs sao tuplas (classificador, discriminador, n_neurons, rank_method, selected_features)
    '''
    
    learners = []
    
    # forcar resetar o mapping a cada execucao de um numero de folds
    if mapping_igual:
        BaseLearner.static_encoder = None
    
    # usar o tamanho das features pra calcular o numero de neuronios
    for classificador, discriminador, bits_endereco, rank_method, selected_features, confusion_matrix_geral in configs_learners:
        totalBits = 0
        for f in selected_features:
            totalBits += tam_features[f]
        #calcular numero de neuronios
        neuronios = totalBits/bits_endereco
        if totalBits % bits_endereco > 0:
            neuronios += 1
        
        learners.append(BaseLearner(discriminador, neuronios, bits_endereco, rank_method, n_folds, confusion_matrix_geral, classificador, mapping_igual, selected_features))
    
    return learners


class Ensemble(object):
    '''
    classe para guardar informacoes e metricas de um ensemble
    '''
    
    def __init__(self, tipo_voto, n_folds, mat_confusao_geral, tipo_erro, tipo_intensidade):
        self.tipo_voto = tipo_voto
        self.tipo_erro = tipo_erro
        self.tipo_intensidade = tipo_intensidade
        
        self.mat_confusao_geral = mat_confusao_geral
        self.mat_confusao = util.ConfusionMatrix()
        self.mat_confusao_folds = []
        for _ in xrange(n_folds):
            self.mat_confusao_folds.append(util.ConfusionMatrix())
        
        self.combined_votes = None
        self.votos = None
        self.intensidades = None
        self.pesos_learners = None
        
        self.agregador = None
        
        self.label = "Ensemble voting= "+tipo_voto+"  tipo_erro= "+str(self.tipo_erro)+"  tipo_intensidade= "+str(self.tipo_intensidade)
        
    def inicia_agregador(self, n_classes=2):
        self.agregador = compo.VotingAggregator(self.votos, self.intensidades, len(self.votos), len(self.votos[0]), n_classes, vote=self.tipo_voto,
                                                      weights=self.pesos_learners)
    
    def predict(self):
        self.combined_votes = self.agregador.predict()
        
    def inicia_votos_e_pesos(self, n_learners, n_instancias):
        self.votos = [[0] * n_learners for _ in xrange(n_instancias)]
        self.intensidades = [[0.0] * n_learners for _ in xrange(n_instancias)]
        self.pesos_learners = [0.0]*n_learners        
        
    def guarda_voto(self, learner, instancia, voto, confianca):
        self.votos[int(instancia)][int(learner)] = int(voto)
        self.intensidades[int(instancia)][int(learner)] = float(confianca)
        
    def guarda_peso(self, learner, peso):
        self.pesos_learners[learner] = peso
        

def cria_matriz_confusao_geral_ensemble():
    matrizes = {}
    mat_boost = {}
    mat_bagging = {}
    
    tipos_voto = ['majority', 'weightedClassifiers']
    tipos_erro = ['nenhum', 'porcentagem', 'proporcao', 'proporcaoDiferenca', 'ativacao', 'porcentagem+ativacao', 'proporcao+ativacao', 'proporcaoDiferenca+ativacao']
    tipos_intensidade = ['nenhum', 'porcentagem', 'proporcao', 'proporcaoDiferenca', 'ativacao', 'porcentagem+ativacao', 'proporcao+ativacao', 'proporcaoDiferenca+ativacao']
    
    for tipo_voto in tipos_voto:
        for tipo_erro in tipos_erro:
            if tipo_voto == 'majority' and tipo_erro != 'nenhum':
                continue
            else:
                for tipo_intensidade in tipos_intensidade:
                    mat_boost[tipo_voto+tipo_erro+tipo_intensidade] = util.ConfusionMatrix()
                    mat_bagging[tipo_voto+tipo_erro+tipo_intensidade] = util.ConfusionMatrix()
    
    matrizes['AdaBoost'] = mat_boost
    matrizes['Bagging'] = mat_bagging
    
    return matrizes
