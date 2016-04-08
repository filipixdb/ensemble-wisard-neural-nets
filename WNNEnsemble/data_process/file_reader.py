'''
@author: filipi
'''

import random as rnd
import pandas as pd

import wann.util as util
from itertools import combinations

def read_and_shuffle_dataset(name):
    try:
    #data = open(data)
# Embaralhar logo as instancias, pra nao precisar fazer no make_folds
        with open(name,'r') as source:
            data = [ (rnd.random(), line) for line in source ]
        
        data.sort()
        
        with open(name+'tmp_file','w') as target:
            for _, line in data:
                target.write( line )
    except:
        pass
        
    try:
        data = open(name+'tmp_file')
    except:
        pass
    return data

def read_dataset(name):
    try:
        data = open(name)
    except:
        pass
    return data

def le_entradas(name, tam_features, shuffle=True):

    # Ler sem perder os zeros
    mapa = {}
    for k in range(21):
        mapa[k] = lambda x: str(x)


    # criar as especificacoes das colunas
    colspecs = []
    colspecs.append((0, 1))
    inicio_feature = 2
    for tam in tam_features:
        colspecs.append( (inicio_feature, (inicio_feature+tam)) )
        inicio_feature += tam

    
    # Ler o arquivo
    df = pd.read_fwf(name, colspecs=colspecs, header=None, index_col=None, converters = mapa)

    classes = df[0]

    primeiro = True
    for z in xrange(len(tam_features)):# em cada atributo
        if primeiro:
            aux = df[z+1]
            primeiro=False
        else:
            aux+=df[z+1]
    
    # juntar as classes e as representacoes
    entradas = pd.concat([classes, aux], axis=1)
    
    entradas_list = entradas.to_records()#[[0],[1]]
    if shuffle:
        return rnd.sample(entradas_list, len(entradas_list))
    else:
        return entradas_list.tolist()
    

def le_informacoes_OLD(arquivo):
    with open(arquivo) as infos:
        n_classes = int(infos.readline())
        temp = infos.readline()
        tamanhos = list(int(x) for x in temp[:-1].split(','))
        n_params = int(infos.readline())
        temp = infos.readline()
        list_n_folds = list(int(x) for x in temp[:-1].split(','))
    return n_classes, tamanhos, n_params, list_n_folds


def le_informacoes(arquivo):
    with open(arquivo) as infos:
        
        temp = infos.readline()
        temp = list(x for x in temp[:-1].split('='))
        n_classes = int(temp[1])
        
        temp = infos.readline()
        temp = list(x for x in temp[:-1].split('='))
        temp = temp[1]
        tamanhos = list(int(x) for x in temp.split(','))

        temp = infos.readline()
        parametrizacoes = []
        while True:
            temp = infos.readline()
            if (temp[:-1] == 'fim') or (temp == 'fim'):
                break
            else:
                parametrizacoes.append(temp[:-1])
        
    return n_classes, tamanhos, parametrizacoes


def le_parametros_OLD(arquivo):
    configs_single_learners = []
    configs_base_learners = []
    with open(arquivo) as params:
        for line in params:
            learner, classificador, discriminador, neuronios, resposta, features = line.split(',')
            features = list(int(x) for x in features[:-1].split('-'))
            config = (classificador, discriminador, int(neuronios), resposta, features, util.ConfusionMatrix())
            if learner == "single_learner":
                configs_single_learners.append(config)
            elif learner == "base_learner":
                configs_base_learners.append(config)
    return configs_single_learners, configs_base_learners






def le_parametros(arquivo):
    with open(arquivo) as params:
        configs_single_learners = []
        configs_base_learners = []
        
        temp = params.readline()
        temp = temp[:-1].split('=')
        n_folds = int(temp[1])
        
        temp = params.readline()
        temp = temp[:-1].split('=')
        n_execucoes = int(temp[1])
        
        temp = params.readline()
        while True:
            temp = params.readline()
            if (temp == 'params_ensemble:') or (temp[:-1] == 'params_ensemble:'):
                break
            else:
                classificador, discriminador, neuronios, resposta, features = temp.split(',')
                features = list(int(x) for x in features[:-1].split('-'))
                config = (classificador, discriminador, int(neuronios), resposta, features, util.ConfusionMatrix())
                configs_single_learners.append(config)
        
        temp = params.readline()
        temp = temp[:-1].split('=')
        tamanho_treino_bagging = float(temp[1])
        
        temp = params.readline()
        temp = temp[:-1].split('=')
        tamanho_treino_boost = float(temp[1])
        
        temp = params.readline()
        temp = temp[:-1].split('=')
        if temp[1] == 'true':
            com_repeticao_bagging = True
        else:
            com_repeticao_bagging = False
        
        temp = params.readline()
        temp = temp[:-1].split('=')
        if temp[1] == 'true':
            com_repeticao_boost = True
        else:
            com_repeticao_boost = False
                
        temp = params.readline()
        temp = temp[:-1].split('=')
        if temp[1] == 'true':
            mesmo_mapping_bagging = True
        else:
            mesmo_mapping_bagging = False
        
        temp = params.readline()
        temp = temp[:-1].split('=')
        if temp[1] == 'true':
            mesmo_mapping_boost = True
        else:
            mesmo_mapping_boost = False
        
        
        temp = params.readline()
        temp = temp[:-1].split('=')
        if temp[1] == 'explicito':
            n_learners = None
            while True:
                temp = params.readline()
                if (temp == 'fim') or (temp[:-1] == 'fim'):
                    break
                else:
                    classificador, discriminador, neuronios, resposta, features = temp.split(',')
                    features = list(int(x) for x in features[:-1].split('-'))
                    config = (classificador, discriminador, int(neuronios), resposta, features, util.ConfusionMatrix())
                    configs_base_learners.append(config)
        else:
            # ler os parametros e depois sortear as configs dos base learners
            temp = params.readline()
            temp = temp[:-1].split('=')
            temp = temp[1]
            ranking_features = list(int(x) for x in temp.split(','))
            
            temp = params.readline()
            temp = temp[:-1].split('=')
            features_fixas = int(temp[1])
            
            temp = params.readline()
            temp = temp[:-1].split('=')
            features_opcionais = int(temp[1])
            
            temp = params.readline()
            temp = temp[:-1].split('=')
            features_sorteadas = int(temp[1])
            
            temp = params.readline()
            temp = temp[:-1].split('=')
            n_learners = int(temp[1])
            
            temp = params.readline()
            temp = temp[:-1].split('=')
            classificador, discriminador, neuronios, resposta = temp[1].split(',')
            
            combinacoes_features = gera_combinacoes_features(ranking_features, features_fixas, features_opcionais, features_sorteadas)
            for combinacao in combinacoes_features:
                config = (classificador, discriminador, int(neuronios), resposta, combinacao, util.ConfusionMatrix())
                configs_base_learners.append(config)
                
        return n_folds, n_execucoes, tamanho_treino_bagging, tamanho_treino_boost, com_repeticao_bagging, com_repeticao_boost, mesmo_mapping_bagging, mesmo_mapping_boost, configs_single_learners, configs_base_learners, n_learners



def gera_combinacoes_features(ranking, fixas, opcionais, sorteadas):
    features_fixas = list(int(x) for x in ranking[0:fixas])
    features_opcionais = list(int(x) for x in ranking[fixas:(fixas+opcionais)])
    combinacoes = list(combinations(features_opcionais, sorteadas))
    
    resultado = []
    for combinacao in combinacoes:
        temp = features_fixas[:]
        temp.extend(list(combinacao))
        resultado.append(temp)
    
    return resultado
