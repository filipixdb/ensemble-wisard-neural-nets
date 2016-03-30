'''
Fazer o encoding do german dataset (ja convertido pra numerico)
'''

import data_process.encoding as enc
import pandas as pd
from _functools import partial

# ler o arquivo
file_name = "files/new_german_TEMP.data"
data = pd.read_csv(file_name, delim_whitespace=True, header=None, names=range(21))


# separar os tipos de atributos
ordinais = set([1, 3, 6, 7, 10, 12])
numericos = set([2, 5, 8, 11, 13, 16, 18])
nao_ordinais = set([4, 9, 14, 15, 17, 19, 20])
binarios = set([19, 20])
max_values = [3, 72, 4, 9, 18424, 4, 4, 4, 3, 2, 4, 3, 75, 2, 2, 4, 3, 2, 1, 1]
min_values = [0, 4, 0, 0, 250, 0, 0, 1, 0, 0, 1, 0, 19, 0, 0, 1, 0, 1, 0, 0]



# analisar os valores das features
for feature in range(20):
    if feature+1 in ordinais.union(nao_ordinais):
        matriz = []
        for i in range(max_values[feature] +1):
            matriz.append([0.0, 0.0, 0])
        for numero, instancia in enumerate(data[feature+1]):
            matriz[data[feature+1][numero]][data[0][numero]] += 1
            matriz[data[feature+1][numero]][2] += 1
            
        print "Feature ", feature
        for i in range(max_values[feature] +1):
            print "Valor ", i, ":    ", matriz[i][0], "    ", matriz[i][1], "    Total: ", matriz[i][2], "    P/N: ", matriz[i][0]/matriz[i][1]





# criar o encoder de cada coluna unaria
encoders_unarios = {}
bits = 24
for i in range(20):
    if (i+1) in ordinais.union(numericos):
        encoder = enc.UnaryEncoder(min_values[i], max_values[i], bits)
        encoders_unarios[i+1] = encoder

# funcao que aplica encoding unario
def encodeUnario(x, column):
    if column in encoders_unarios:
        pre = int(x[column])
        encoder = encoders_unarios[column]
        pos = encoder([pre])
        return pos

# aplicar o encoding nas colunas numericas e ordinais
for i in range(20):
    if (i+1) in ordinais.union(numericos):
        data[i+1] = data.apply(partial(encodeUnario, column=(i+1)), axis=1)






def hamdist(str1, str2):
        diffs = 0
        for ch1, ch2 in zip(str1, str2):
                if ch1 != ch2:
                        diffs += 1
        return diffs






# criar o encoder de cada coluna nao ordinal
encoders_qualitativos = {}
bits = 24
for i in range(20):
    if (i+1) in nao_ordinais:
        encoder = enc.QualitativeEncoder(bits)
        # colocar um dummie pra ser a distancia diferente
        # para o atributo 4, sera o valor 'others'
        # corrigir para nao chamar para as features binarias
        if (i+1) == 4:
            encoder([9])
            
        # colocar mais distante o valor None
        if (i+1) == 14:
            encoder([2])
        
        print "Feature: ", i+1
        
        
        # definir logo os encodes, pra ordem das instancias nao interferir
        for aux in range(max_values[i]+1):
#            encoder([i])#TODO: esta bugado!!! deve chamar para os valores aux e nao para i
            encoder([aux])


        for aux in range(max_values[i]+1):
            print "  ", encoder([aux]), "  -> ", aux
            for aux2 in range(max_values[i]+1):
                print "    ", aux, " dist ", aux2, " -> ", hamdist(encoder([aux]), encoder([aux2]))

            
        
        encoders_qualitativos[i+1] = encoder



# funcao que aplica encoding qualitativo
def encodeQualitativo(x, column):
    if column in encoders_qualitativos:
        pre = int(x[column])
        encoder = encoders_qualitativos[column]
        pos = encoder([pre])
        return pos

# aplicar o encoding nas colunas nao ordinais
for i in range(20):
    if (i+1) in nao_ordinais:
        data[i+1] = data.apply(partial(encodeQualitativo, column=(i+1)), axis=1)

'''
nominais com 8 bits
 ..9 valores -> uma distancia 8, o resto 4
 10 valores -> duas distancias 8, resto 4
 11 valores -> tres distancias 8, resto 4
'''

# concatenar os encodes
classes = data[0]
instancias = data[1]
for coluna in range(19):
    instancias = instancias+data[coluna+2]





# escrever arquivo

encoded_data = pd.concat([classes, instancias], axis=1)
output_file_name = "files/encoded_german_24_bits_TEMP.data"
encoded_data.to_csv(output_file_name, header=False, index=False, sep=" ")
