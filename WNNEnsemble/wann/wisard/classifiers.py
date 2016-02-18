import collections as cl
import itertools as it

from llist import dllist

from discriminators import Discriminator
from neurons import MultiValueNeuron
import wann.util as util
import wann.encoding as encoding


class WiSARDLikeClassifier(object):
    '''The superclass of all WiSARD-like classifiers.

    This should be used as a template to any classifier implementation, as an
    abstract class, but no method is indeed required to be overriden.
    '''

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('This class is abstract. Derive it.')

    def record(self, observation, class_):
        '''Record the provided observation, relating it to the given class.'''
        raise NotImplementedError('This method is abstract. Override it.')

    def answers(self, observation, class_=None):
        '''Returns how similar the observation is to the known classes.

        By default, a dictionary with the classes labels as keys and the
        respective similarities between observation and classes as values is
        returned.

        Parameters
            observation: the observation in which the answers must be based
            class_: if given, only the similarity wrt this class is returned
        '''

        raise NotImplementedError('This method is abstract. Override it.')

    def counts(self, observation, class_=None):
        '''Returns a description of observation similarity to known classes.

        The description of the similarity must take into account the number of
        times the observation addresses were previously recorded.
        
        By default, a dictionary with the classes labels as keys and theirs
        respective similarity descriptions as values is returned.

        Parameters
            observation: the observation in which the answers must be based
            class_: if given, only the answer wrt this class is returned
        '''

        raise NotImplementedError('This method is abstract. Override it.')

    def remove_class(self, class_):
        raise NotImplementedError('This method is abstract. Override it.')


class WiSARD(WiSARDLikeClassifier):
    def __init__(self, discriminator=Discriminator):
        '''Inits a WiSARD classifier using the provided arguments.

        Parameters
            discriminator: the discriminator which will be used to learn about
                each class to be presented. The argument must be callable,
                returning a Discriminator-like object.
        '''

        self.discriminators = cl.defaultdict(discriminator)

    def record(self, observation, class_):
        self.discriminators[class_].record(observation)

# retorna um monte de respostas dos discriminadores para uma observation
    def answers(self, observation, class_=None):
        if class_ is not None:
            return self.discriminators[class_].answer(observation)

        return {class_: dscrmntr.answer(observation)
            for class_, dscrmntr in self.discriminators.viewitems()}

    def counts(self, observation, class_=None):
        if class_ is not None:
            return self.discriminators[class_].counts(observation)

        return {class_: dscrmntr.counts(observation)
            for class_, dscrmntr in self.discriminators.viewitems()}

    def bleach(self, threshold):
        for d in self.discriminators:
            self.discriminators[d].bleach(threshold)

    def remove_class(self, class_):
        del self.discriminators[class_]


class MultiClassDiscriminator(WiSARDLikeClassifier):
    def __init__(self, neuron=MultiValueNeuron):
        self.neuron_factory = neuron
        self.neurons = None

    def record(self, observation, class_):
        if self.neurons is None:
            self.neurons = [self.neuron_factory() for _ in observation]

        for address, neuron in it.izip(observation, self.neurons):
            neuron.record(address, class_)

    def answers(self, observation, class_=None):
        the_answers = (neuron.answer(address, class_)
            for address, neuron in it.izip(observation, self.neurons))

        if class_ is not None:
            return sum(the_answers)

        return cl.Counter(class_
            for answer in the_answers for class_ in answer)

    def counts(self, observation, class_=None):
        the_freqs = (neuron.count(address, class_)
            for address, neuron in it.izip(observation, self.neurons))

        if class_ is not None:
            return sorted(the_freqs)

        counter = cl.defaultdict(list)

        for freqs in the_freqs:
            for a_class_, count in freqs.viewitems():
                counter[a_class_].append(count)

        for f in counter.viewvalues():
            f.sort()

        return counter

    def remove_class(self, class_):
        for neuron in self.neurons:
            neuron.remove_class(class_)


class SelfContainedWiSARD(WiSARDLikeClassifier):
    def __init__(self, nmbr_neurons):
        discriminator = lambda: [
            cl.defaultdict(int) for _ in it.repeat(None, nmbr_neurons)]

        self.dscrmntrs = cl.defaultdict(discriminator)

    def record(self, observation, class_):
        for address, neuron in it.izip(observation, self.dscrmntrs[class_]):
            neuron[address] += 1

    def answers(self, observation, is_sorted=False, class_=None,
            normalized=False):
        if class_ is not None:
            answer = sum(address in neuron for address, neuron in
                it.izip(observation, self.dscrmntrs[class_]))

            if normalized:
                return answer * 1. / len(self.dscrmntrs[class_])

            return answer

        raise Exception


    def bleach(self, threshold):
        for d in self.dscrmntrs.values():
            for neuron in d:
                for address, value in neuron.items():
                    if value <= threshold:
                        del neuron[address]
                    else:
                        neuron[address] -= threshold


def ClusWiSARDfactory(parent=WiSARD):
    return type('ClusWiSARDsubclass', (ClusWiSARD, parent), {})


class ClusWiSARD(WiSARDLikeClassifier):
    def __getstate__(self):
        global ClusWiSARDsubclass
        ClusWiSARDsubclass = self.__class__
        dict_ = self.__dict__.copy()
        dict_['clusters'] = list(dict_['clusters'])
        return dict_

    def __setstate__(self, dict_):
        dict_['clusters'] = dllist(dict_['clusters'])
        self.__class__ = ClusWiSARDfactory()
        self.__dict__.update(dict_)

    def __init__(self, min_similarity, expected_absorptions, max_clusters=None,
            *args, **kwargs):
        super(ClusWiSARD, self).__init__(*args, **kwargs)
        self.min_similarity = min_similarity
        self.expected_absorptions = expected_absorptions
        self.clusters = dllist()
        self.next_id = 0
        self.max_clusters = max_clusters
        self.infos = {}

    def record(self, observation, class_=None):
        for node, cluster in util.iter_llist_items(self.clusters):
            info = self.infos[cluster]

            if info.class_ is None or class_ is None or info.class_ == class_:
                threshold = info.absorptions * 1.
                threshold /= self.expected_absorptions
                threshold += self.min_similarity

                answer = self.answers(observation, cluster)
                if answer >= len(observation) * min(threshold, 1.):
                    self.clusters.remove(node)
                    break
        else:
            if len(self.infos) == self.max_clusters:
                to_remove = min(
                    (v.absorptions, k) for k, v in self.infos.viewitems())
                self.remove_class(to_remove[1])

            cluster, self.next_id = self.next_id, self.next_id + 1
            info = self.infos[cluster] = util.ClusterInfo()

        if class_ is not None:
            info.class_ = class_

        super(ClusWiSARD, self).record(observation, cluster)
        self.clusters.appendleft(cluster)
        info.absorptions += 1

        return cluster

    def remove_class(self, cluster):
        try:
            node, cluster = cluster, cluster.value
            return_value = node.next
        except AttributeError:
            node = self.clusters.first
            while node:
                if node.value == cluster:
                    break
                node = node.next

            return_value = None

        super(ClusWiSARD, self).remove_class(cluster)
        self.clusters.remove(node)
        del self.infos[cluster]

        return return_value

    def discard(self, threshold):
        node = self.clusters.first
        while node:
            if self.infos[node.value].absorptions <= threshold:
                node = self.remove_class(node)
            else:
                node = node.next


class ClusWiSARDsubclass(ClusWiSARD):
    pass
