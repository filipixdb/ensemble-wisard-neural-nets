'''
@author: filipi
'''
import math

# Metodos para avaliar a performance dos classificadores e ensembles

#calcular a media, variancia e desvio
def media_variancia_desvio(valores):
    soma = 0.0
    for valor in valores:
        soma+=float(valor)

    media = soma/len(valores)
    
    variancia = 0.0
    for valor in valores:
        variancia += math.pow((float(valor) - media), 2)
    variancia = variancia/len(valores)
    
    desvio = math.sqrt(variancia)

    resultado = {}
    resultado['soma'] = soma
    resultado['media'] = media
    resultado['variancia'] = variancia
    resultado['desvio'] = desvio

    return resultado