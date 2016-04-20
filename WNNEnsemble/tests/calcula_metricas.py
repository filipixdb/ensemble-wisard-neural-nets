'''
Created on 18 de abr de 2016

@author: filipi
'''

import sys
import ast
sys.path.append("../")
import metrics.performance as metr


def main(nome_arquivo):
    
    valores_bagging_acc = []
    valores_bagging_f1 = []
    valores_bagging_custo = []
    
    valores_boost_acc = []
    valores_boost_f1 = []
    valores_boost_custo = []
    
    
    with open(nome_arquivo) as arq:
        linha = arq.readline()
        while (linha[4:-1] != 'Bagging\r') and (linha[4:-1] != 'Bagging'):
            linha = arq.readline()
        #lendo o Bagging
        while linha[6:6+9] != 'Ensembles':
            linha = arq.readline()
        #lendo os ensembles do bagging
        while (linha[4:-1] != 'AdaBoost\r') and (linha[4:-1] != 'AdaBoost'):
            if linha[15:15+8] == 'accuracy':
                linha = linha[12:-1]
                lista = ast.literal_eval(linha)
                
                _, acc = (lista[0])
                _, f1 = (lista[1])
                _, custo = (lista[2])
                
                valores_bagging_acc.append(acc)
                valores_bagging_f1.append(f1)
                valores_bagging_custo.append(custo)
            linha = arq.readline()
        #lendo Boost
        while linha[6:6+9] != 'Ensembles':
            linha = arq.readline()
        #lendo os ensembles do boost
        while (linha[4:-1] != 'AdaBoost\r') and (linha != ''):
            if linha[15:15+8] == 'accuracy':
                linha = linha[12:-1]
                lista = ast.literal_eval(linha)
                
                _, acc = lista[0]
                _, f1 = lista[1]
                _, custo = lista[2]
                
                valores_boost_acc.append(acc)
                valores_boost_f1.append(f1)
                valores_boost_custo.append(custo)
            linha = arq.readline()
        #leu tudo
    
    metricas_bagging_acc = metr.media_variancia_desvio(valores_bagging_acc)
    metricas_bagging_f1 = metr.media_variancia_desvio(valores_bagging_f1)
    metricas_bagging_custo = metr.media_variancia_desvio(valores_bagging_custo)
    
    metricas_boost_acc = metr.media_variancia_desvio(valores_boost_acc)
    metricas_boost_f1 = metr.media_variancia_desvio(valores_boost_f1)
    metricas_boost_custo = metr.media_variancia_desvio(valores_boost_custo)

    
    print "Resultados:"
    print "  Bagging"
    print "    Acc"
    print "      maior = ", max(valores_bagging_acc)
    print "      media = ", metricas_bagging_acc['media']
    print "      variancia = ", metricas_bagging_acc['variancia']
    print "      desvio = ", metricas_bagging_acc['desvio']
    print "    F1"
    print "      maior = ", max(valores_bagging_f1)
    print "      media = ", metricas_bagging_f1['media']
    print "      variancia = ", metricas_bagging_f1['variancia']
    print "      desvio = ", metricas_bagging_f1['desvio']
    print "    Custo"
    print "      menor = ", min(valores_bagging_custo)
    print "      media = ", metricas_bagging_custo['media']
    print "      variancia = ", metricas_bagging_custo['variancia']
    print "      desvio = ", metricas_bagging_custo['desvio']
    print "  Boost"
    print "    Acc"
    print "      maior = ", max(valores_boost_acc)
    print "      media = ", metricas_boost_acc['media']
    print "      variancia = ", metricas_boost_acc['variancia']
    print "      desvio = ", metricas_boost_acc['desvio']
    print "    F1"
    print "      maior = ", max(valores_boost_f1)
    print "      media = ", metricas_boost_f1['media']
    print "      variancia = ", metricas_boost_f1['variancia']
    print "      desvio = ", metricas_boost_f1['desvio']
    print "    Custo"
    print "      menor = ", min(valores_boost_custo)
    print "      media = ", metricas_boost_custo['media']
    print "      variancia = ", metricas_boost_custo['variancia']
    print "      desvio = ", metricas_boost_custo['desvio']
    
    

if __name__ == '__main__':
    main(*sys.argv[1:])
