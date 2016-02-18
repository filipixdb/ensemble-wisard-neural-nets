'''
@author: filipi
'''

# Metodos para compor a resposta final
# fazer um metodo que compoe os votos com peso
# receber um array[][] de [n_instancias][n_classificadores]
# onde cada celula eh um (voto, peso)
# o peso eh montado por fora de acordo com:
#    performance de cada classificador
#    pertinencia ou confianca de cada classificador
#    peso das features usadas por cada classificador
#    pesos iguais (maioria dos votos)
# retornar um array de (classificacao)
#
# depois posso usar o classifiers pra mostrar quais classificadores foram ouvidos na escolha final,
# pra ver tbm quantas vezes um classificador pontuou e etc


import numpy as np


class VotingAggregator(object):
    """ Compoe os varios votos de acordo com os pesos
    
    possiveis valores vote: majority, weightedClassifiers, weightedInstances
    
    """

    def __init__(self, votes, n_instances, n_classifiers, n_classes, classifiers=None,
                 vote='majority', weights=None):
        
        self.votes = votes
        self.classifiers = classifiers
        self.vote = vote
        self.weights = weights
        self.n_classes = n_classes
        
        self.n_instances = n_instances
        self.n_classifiers = n_classifiers
                
        # cria a matriz com a quantidade de votos
        self.vote_count = np.zeros((self.n_instances, self.n_classes))
        
        # cria array de respostas
        self.combined_votes = [0] * n_instances

    def predict(self):
        
        dim1, dim2 = self.n_instances, self.n_classifiers
            
        if self.vote == 'majority':
            for i in range(dim1):
                for j in range(dim2):
                    prediction = int(self.votes[i][j])
                    self.vote_count[i,prediction] += 1
                # preencher a resposta final considerando sorteio para empate
                max_value_index = np.where(self.vote_count[i,:] == self.vote_count[i,:].max())
                max_value_index = max_value_index[0]
#                if len(max_value_index)>1:
#                    print ('DEBUG: empate no majority voting -- '), max_value_index
                self.combined_votes[i] = int(np.random.choice(max_value_index))
        elif self.vote == 'weightedClassifiers': # use weights como aray dim1
            media_pesos = sum(self.weights)/len(self.weights)
            for i in range(dim1):
                for j in range(dim2):
                    prediction = int(self.votes[i][j])
                    peso = (self.weights[j]/media_pesos)
                    peso = peso**5
                    self.vote_count[i,prediction] += (1*peso)
                # preencher a resposta final considerando sorteio para empate
                max_value_index = np.where(self.vote_count[i,:] == self.vote_count[i,:].max())
                max_value_index = max_value_index[0]
#                if len(max_value_index)>1:
#                    print ('DEBUG: empate no majority voting -- '), max_value_index
                self.combined_votes[i] = int(np.random.choice(max_value_index))
        else: # use weights como array[instancias][classificadores]
            for i in range(dim1):
                for j in range(dim2):
                    prediction = self.votes[i][j]
                    self.vote_count[i][prediction] += (1*self.weights[i][j])
                # preencher a resposta final considerando sorteio para empate
                max_value_index = np.where(self.vote_count[i,:] == self.vote_count[i,:].max())
                max_value_index = max_value_index[0]
                if len(max_value_index)>1:
                    print ('DEBUG: empate no majority voting -- '), max_value_index
                self.combined_votes[i] = int(np.random.choice(max_value_index))
        

'''    
    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        
        if not deep:
            return super(MajorityVoteClassifier,
                         self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in\
                    six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(
                                                step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out
    
'''