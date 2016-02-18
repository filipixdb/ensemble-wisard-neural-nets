'''
@author: filipi
'''

import random as rnd

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

