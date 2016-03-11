'''
@author: filipi
'''

# Metodos para preprocessamento (selecionar features, instancias, corrigir noise, blank etc)

# fazer aqui os metodos que ajudam a criar as versoes do dataset com algumas features a menos


import pandas as pd
from pandas.io.parsers import read_fwf

import random as rnd
    
def le_features(file_name, selected_features):
    '''
    Chamada dummie da funcao
    selected_features = [[1, 20], [7, 9, 13]]
    le_features('encoded_german_menor_ainda.data', selected_features)
    '''

    print "FILE NAME: ", file_name

    '''
    colspecs = [(0, 1), (2, 10),  (10, 18),  (18, 26),
            (26, 34),  (34, 42),  (42, 50),  (50, 58),
            (58, 66),  (66, 74),  (74, 82),  (82, 90),
            (90, 98),  (98, 106),  (106, 114),
            (114, 122),  (122, 130),  (130, 138),
            (138, 146),  (146, 154),  (154, 162)]
    '''
    
    colspecs = [(0, 1), (2, 18), (18, 34), (34, 50 ),
                (50, 66), (66, 82), (82, 98), (98, 114),
                (114, 130), (130, 146), (146, 162),
                (162, 178), (178, 194), (194, 210),
                (210, 226), (226, 242), (242, 258),
                (258, 274), (274, 290), (290, 306), (306, 322)]

    # Vai armazenar os nomes dos arquivos
    data = {}

    #classes = df[0]
    for i, comb in enumerate(selected_features):# em cada classificador
        # Ler sem perder os zeros
        mapa = {}
        for k in range(21):
            mapa[k] = lambda x: str(x)
        df = read_fwf(file_name, colspecs=colspecs, header=None, index_col=None, converters = mapa)
        classes = df[0]

        primeiro = True
        for z in comb:# em cada atributo
            if primeiro:
                aux = df[z]
                primeiro=False
            else:
                aux+=df[z]
        #tag_arq = ''.join(str(x).zfill(2) for x in comb)
        chave = str(i)
        
        encoded_file_name = "files/"+chave
        
        encoded = pd.concat([classes, aux], axis=1)
        encoded.to_csv(encoded_file_name, header=False, index=False, sep=" ")
        
        data[chave] = encoded_file_name
                
    return data


def sorteia_features(qnt_tipos_neuro=4, tipos_classi=3, qnt_por_classi=2, total_feat=20, qnt_feat=12):
    total_sortear = qnt_tipos_neuro * tipos_classi * qnt_por_classi
    sorteados = []
    for _ in range(total_sortear):
        features = rnd.sample(range(total_feat), qnt_feat)
        features = [x+1 for x in features]
        features.sort()
        sorteados.append(features)
    return sorteados
