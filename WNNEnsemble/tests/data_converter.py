'''
Converter o german dataset para um formato numerico
'''

import pandas as pd
from _functools import partial

# ler o arquivo
file_name = "files/german.data"
data = pd.read_csv(file_name, delim_whitespace=True, header=None, names=range(21))
# colocar a classe como primeira coluna
data = data[[20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]

# remover as labels 'Axx'
def removeAx(x, column):
    if column > 9:
        offset = 3
    else:
        offset = 2
    return int(x[column][offset:])
colunas = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
for i in colunas:
    data[i] = data.apply(partial(removeAx, column=i), axis=1)


# colocar todos os qualitativos pra comecar em 0
def decrementaValores(x, column):
    return int(x[column]-1)
colunas = [20, 0, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
for i in colunas:
    data[i] = data.apply(partial(decrementaValores, column=i), axis=1)



# decrementar A48, A49, A10...
def decrementaA4(x):
    if int(x[3]) > 7:
        return int(x[3]-1)
    else:
        return int(x[3])
data[3] = data.apply(decrementaA4, axis=1)


# descobrir range dos numericos -> headers [1, 4, 7, 10, 12, 15, 17]
# na verdade nem eh necessario pq o encoding consegue encaixar num numero x de bits
def valorMaximo(x, column):
    instancia = x.loc[x[column].idxmax()]
    return instancia[column]
def valorMinimo(x, column):
    instancia = x.loc[x[column].idxmin()]
    return instancia[column]

# para os numericos
#colunas = [1, 4, 7, 10, 12, 15, 17]

# para os ordinais
#colunas = [0, 2, 5, 6, 9, 11, 14, 16]

# para os nao ordinais
#colunas = [3, 8, 13, 18, 19]

#for i in colunas:
#    maximo = valorMaximo(data, i)
#    minimo = valorMinimo(data, i)
#    print ("Col %d:  Max: %d   Min %d" % (i, maximo, minimo))


'''
Range dos atributos numericos
Col 1:  Max: 72   Min 4 (7 bits)
Col 4:  Max: 18424   Min 250 (16 bits)
Col 7:  Max: 4   Min 1
Col 10:  Max: 4   Min 1
Col 12:  Max: 75   Min 19 (7 bits)
Col 15:  Max: 4   Min 1
Col 17:  Max: 2   Min 1
'''    

'''
Range dos atributos ordinais
Col 0:  Max: 3   Min 0
Col 2:  Max: 4   Min 0
Col 5:  Max: 4   Min 0
Col 6:  Max: 4   Min 0
Col 9:  Max: 2   Min 0
Col 11:  Max: 3   Min 0
Col 14:  Max: 2   Min 0
Col 16:  Max: 3   Min 0
'''

'''
Possiveis valores dos atributos nao ordinais
Col 3:  Max: 9   Min 0
Col 8:  Max: 3   Min 0
Col 13:  Max: 2   Min 0
Col 18:  Max: 1   Min 0
Col 19:  Max: 1   Min 0
'''

'''
Tipos de atributos (numero do atributo e nao da coluna do dataframe)
ordinais [1, 3, 6, 7, 10, 12, 17]
numericos [2, 5, 8, 11, 13, 16, 18]
nao ordinais [4, 9, 14, 15, 19, 20]
'''


# Corrigir a ordem dos ordinais
#   atributo 1 -> +1 mod 4
#   atributo 6 -> +1 mod 5
#   atributo 15 -> +2 mod 3

def corrigeOrdinalA1(x):
    valor = int(x[0])
    valor += 1
    return (valor % 4)
def corrigeOrdinalA6(x):
    valor = int(x[5])
    valor += 1
    return (valor % 5)
def corrigeOrdinalA10(x):
    valor = int(x[9])
    if valor == 0:
        valor = 1
    elif valor == 1:
        valor = 0
    elif valor == 2:
        pass
    return valor
def corrigeOrdinalA15(x):
    valor = int(x[14])
    valor += 2
    return (valor % 3)

#na verdade do jeito original esta melhor
#data[0] = data.apply(corrigeOrdinalA1, axis=1)
data[5] = data.apply(corrigeOrdinalA6, axis=1)
data[9] = data.apply(corrigeOrdinalA10, axis=1)
data[14] = data.apply(corrigeOrdinalA15, axis=1)


# escrever
'''
output_file_name = "files/new_german_TEMP.data"
data.to_csv(output_file_name, header=False, index=False, na_rep=" ", sep=" ")
'''