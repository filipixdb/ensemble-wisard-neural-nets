'''
@author: filipi
'''

import random as rnd
import pandas as pd

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
    
