'''
@author: filipi
'''
import itertools as it
import random as rnd

# Fazer bootstrap dos datasets

# fazer a funcao do bagging com repeticao
def bootstrap_wr(data, percent):
    aux = it.chain(*data)
    treino = [x for x in aux]
    return [treino[i] for i in sample_wr(xrange(len(treino)), int(len(treino)*percent))]

# fazer funcao de bagging sem repeticao

def bootstrap(data, percent):
    aux = it.chain(*data)
    treino = [x for x in aux]
    return [treino[i] for i in rnd.sample(xrange(len(treino)), int(len(treino)*percent))]

# com repeticao

def sample_wr(population, k):
    "Chooses k random elements (with replacement) from a population"
    n = len(population)
    _random, _int = rnd.random, int  # speed hack 
    result = [None] * k
    for i in xrange(k):
        j = _int(_random() * n)
        result[i] = population[j]
    return result
