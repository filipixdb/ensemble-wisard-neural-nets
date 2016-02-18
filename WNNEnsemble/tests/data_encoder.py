'''
@author: filipi
'''
'''

Fazer o encoding do german dataset (ja convertido pra numerico)
'''

import data_process.encoding as enc
import pandas as pd
from _functools import partial

# ler o arquivo
file_name = "files/new_german.data"
data = pd.read_csv(file_name, delim_whitespace=True, header=None, names=range(21))


# separar os tipos de atributos
ordinais = set([1, 3, 6, 7, 10, 12, 15, 17])
numericos = set([2, 5, 8, 11, 13, 16, 18])
nao_ordinais = set([4, 9, 14, 19, 20])
max_values = [3, 100, 4, 9, 20000, 4, 4, 4, 3, 2, 4, 3, 100, 2, 2, 4, 3, 2, 1, 1]

# criar o encoder de cada coluna unaria
encoders_unarios = {}
bits = 8
for i in range(20):
    if (i+1) in ordinais.union(numericos):
        encoder = enc.UnaryEncoder(0, max_values[i], bits)
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



# criar o encoder de cada coluna nao ordinal
encoders_qualitativos = {}
bits = 8
for i in range(20):
    if (i+1) in nao_ordinais:
        encoder = enc.QualitativeEncoder(bits)
        # colocar um dummie pra ser a distancia diferente
        # para o atributo 4, sera o valor 'others'
        # para os outros atributos, sera o valor '9' (nao existente)
        encoder([9])
        
        # definir logo os encodes, pra ordem das instancias nao interferir
        for aux in range(max_values[i]+1):
            encoder([i])
        
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
output_file_name = "files/encoded_german.data"
encoded_data.to_csv(output_file_name, header=False, index=False, na_rep=" ", sep=" ")
