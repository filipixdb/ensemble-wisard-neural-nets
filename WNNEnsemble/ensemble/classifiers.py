'''
@author: filipi
'''

import time

from _functools import partial
import wann.vgram as wann_vgram
import wann.wisard.classifiers as wann_wis_clf
import wann.wisard.discriminators as wann_wis_dsc
import wann.util as util
import wann.crossval as wann_xval
import ensemble.composition as compo

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
    
    def __init__(self, discriminador, n_neurons, rank_method, n_folds, classificador='wisard', selected_features = None):
        '''
        classifier
        label
        rank_method
        n_neurons
        mat_confusao
        mat_confusao_folds
        selected features
        '''
        self.classificador = cria_classificador(classificador, discriminador)
        self.n_neurons = n_neurons
        self.rank_method = rank_method
        self.selected_features = selected_features
        
        self.mat_confusao = util.ConfusionMatrix()
        self.mat_confusao_folds = []
        for _ in xrange(n_folds):
            self.mat_confusao_folds.append(util.ConfusionMatrix)
        
        self.label = discriminador+" - neuronios= "+n_neurons+" - rank= "+rank_method+" - features= "+selected_features
        
    
def cria_classificador(classificador, discriminador):
    
    if discriminador == 'wisard':
        disc = wann_wis_dsc.Discriminator
    if discriminador == 'lottery':
        disc = wann_wis_dsc.LotteryDiscriminator
    if discriminador == 'vgwisard':
        disc = wann_vgram.VGWiSARDDiscriminator
    
    return wann_wis_clf.WiSARD(disc)
    
def cria_base_learners(configs_classificadores, n_folds):
    
    '''
    recebe uma lista de tuplas com as configs dos base learners
    retorna uma lista com os base learners
    '''
    
    base_leaners = []
    
    for classificador, discriminador, n_neurons, rank_method, selected_features in configs_classificadores:
        base_leaners.append(BaseLearner(discriminador, n_neurons, rank_method, n_folds, classificador, selected_features))
    
    return base_leaners


class Ensemble(object):
    '''
    classe para guardar informacoes e metricas de um ensemble
    '''
    
    def __init__(self, tipo_voto, n_folds):
        self.tipo_voto = tipo_voto
        
        self.mat_confusao = util.ConfusionMatrix
        self.mat_confusao_folds = []
        for _ in xrange(n_folds):
            self.mat_confusao_folds.append(util.ConfusionMatrix)
        
        self.label = "Ensemble voting= "+tipo_voto
        
    def inicia_agregador(self, votos, n_classes=2, pesos=None):
        self.agregador = compo.VotingAggregator(votos, len(votos), len(votos[0]), n_classes, self.tipo_voto,
                                                      weights=pesos)
    
    def predict(self):
        self.agregador.predict()