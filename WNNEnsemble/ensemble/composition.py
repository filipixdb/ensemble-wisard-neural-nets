
import numpy as np


class VotingAggregator(object):
    """ Compoe os varios votos de acordo com os pesos
    
    possiveis valores vote: majority, weightedClassifiers, weightedInstances
    
    """

    def __init__(self, votes, confiancas, n_instances, n_classifiers, n_classes, classifiers=None,
                 vote='majority', weights=None):
        
        self.votes = votes
        self.intensidades = confiancas
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

        def preenche_maior_resposta(i):
            # acha a resposta final considerando sorteio para empate
            max_value_index = np.where(self.vote_count[i,:] == self.vote_count[i,:].max())
            max_value_index = max_value_index[0]
            self.combined_votes[i] = int(np.random.choice(max_value_index))
            
        if self.vote == 'majority':
            for i in xrange(self.n_instances):
                for c in xrange(self.n_classifiers):
                    prediction = int(self.votes[i][c])
                    self.vote_count[i,prediction] += self.intensidades[i][c]
                preenche_maior_resposta(i)
        
        elif self.vote == 'weightedClassifiers': # use weights como aray dim1
            for i in xrange(self.n_instances):
                for c in xrange(self.n_classifiers):
                    prediction = int(self.votes[i][c])
                    peso = self.weights[c]
                    self.vote_count[i,prediction] += (self.intensidades[i][c]*peso)
                preenche_maior_resposta(i)
                
        else: # use weights como array[instancias][classificadores]
            for i in xrange(self.n_instances):
                for c in xrange(self.n_classifiers):
                    prediction = self.votes[i][c]
                    self.vote_count[i][prediction] += (self.intensidades[i][c]*self.weights[i][c])
                preenche_maior_resposta(i)
                
        return self.combined_votes
        
