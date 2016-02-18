import itertools as it
import math
import numpy as np
import operator as op

from neurons import DictNeuron, SWNeuron


class Discriminator(object):
    '''The default WiSARD discriminator.'''

    def __init__(self, neuron_factory=DictNeuron):
        '''Inits a Discriminator with the given number of neurons.
        
        The type of neuron to be used can be specified through the neuron
        parameter. The argument must be callable, returning a neuron object.
        '''

        self.neurons = None
        self.neuron_factory = neuron_factory

    def __len__(self):
        return len(self.neurons)

    def range(self):
        '''Returns the logarithm of the number of input patterns known.'''
        return np.log([len(neuron) for neuron in self.neurons]).sum()

    def record(self, observation):
        '''Record the provided observation.

        The observation is expected to be a list of addresses, each of which
        will be recorded by its respective neuron.
        '''

        if self.neurons is None:
            self.neurons = [self.neuron_factory() for _ in observation]

        for address, neuron in it.izip(observation, self.neurons):
            neuron.record(address)

    def answer(self, observation):
        '''Returns how similar the observation is to the stored knowledge.

        The return value is the sum of the answers of each neuron to its
        respective address. This value can be normalized by being divided
        by the number of neurons.
        '''

        return sum(neuron.answer(address)
            for address, neuron in it.izip(observation, self.neurons))

    def counts(self, observation):
        '''Returns how many times the observation addresses were recorded.

        This method is intended to be used to compare discriminators
        answers using bleaching. This way, the counts are sorted
        what permits 'regular' comparisons (cmp(), <, and others).

        '''

        counts = (neuron.count(address)
            for address, neuron in it.izip(observation, self.neurons))

        return sorted(c for c in counts if c)

    def drasiw(self):
        '''Returns how many times each bit was set in the addresses recorded.

        This method return a list of lists 'drasiw', where drasiw[i][j] is
        the number of times the bit j was set in the addresses recorded
        by neuron i.

        '''
        return [neuron.bit_counts() for neuron in self.neurons]

    def intersection_level(self, dscrmntr):
        '''Returns the intersection level between the discriminators.

        This is calculated as the mean intersection level between the
        the n-th neurons of each discriminator, for every possible n.
        '''

        return np.mean([na.intersection_level(nb)
            for na, nb in it.izip(self.neurons, dscrmntr.neurons)])

    def bleach(self, threshold):
        '''Bleach each discriminator neuron by the given threshold.'''

        for n in self.neurons:
            n.bleach(threshold)


class SWDiscriminator(Discriminator):
    def __init__(self, number_neurons, neuron=SWNeuron):
        super(SWDiscriminator, self).__init__(number_neurons, neuron)
        self.max_neuron_len = 0

    def __len__(self):
        return self.max_neuron_len

    def record(self, observation, time):
        for address, neuron in it.izip(observation, self.neurons):
            neuron.record(address, time)
            self.max_neuron_len = max(self.max_neuron_len, len(neuron))

    def dump_old_data(self, threshold):
        for neuron in self.neurons:
            neuron.dump_old_data(threshold)
            self.max_neuron_len = max(self.max_neuron_len, len(neuron))

    def answer(self, observation, time_threshold, normalized=False):
        answer = sum(neuron.answer(address, time_threshold)
            for address, neuron in it.izip(observation, self.neurons))

        if normalized:
            return answer * 1. / len(self.neurons)

        return answer



class LotteryDiscriminator(Discriminator):
    '''A discriminator that works on the relative writing frequencies.

    New attributes of LotteryDiscriminator (compared to Discriminator):
        - 'total_writes', which counts the total number of 'record()'
          performed.

    How LotteryDiscriminator answers to a query:
        1. Let tw = LotteryDiscriminator's total_writes
        2. For each (neuron,address) pair
        2.1. Let c = neuron.count(address)
        2.2. If c == 0
        2.2.1. neuron's answer is (1+tw)**-2
        2.2.2. Else, neuron's answer is c/(1+tw)
        3. Let p be the product of the sequence of each neuron answer
        4. LotteryDiscriminator answer is log(p)

    When c == 0, that is, when neuron's location 'address' was not
    recorded even once its "absolute" answer is 1/(1+tw), which works
    as a penalization factor. Otherwise, the "absolute" answer is
    neuron.count(address).
    '''

    def __init__(self, penalty_factor=1.5, *args, **kwargs):
        super(LotteryDiscriminator, self).__init__(*args, **kwargs)
        self.penalty_factor = penalty_factor
        self.total_writes = 0

    def record(self, observation):
        super(LotteryDiscriminator, self).record(observation)
        self.total_writes += 1

    def answer(self, observation):
        ans = 0.
        misses = 0

        for neuron, address in it.izip(self.neurons, observation):
            c = neuron.count(address)
            if c == 0:
                misses += 1
            else:
                ans += math.log(c)
                
        penalty = math.log(1 + self.total_writes)
        penalty *= 1 + (misses ** self.penalty_factor)

        hits = len(self) - misses

        return (ans/hits if hits else 0) - penalty

    def answer2frequency(self, answer, misses=0):
        penalty = math.log(1 + self.total_writes)
        penalty *= misses ** self.penalty_factor

        return math.exp(answer + penalty)

    def frequency2answer(self, frequency, misses=0):
        penalty = math.log(1 + self.total_writes)
        penalty *= misses ** self.penalty_factor

        return math.log(frequency) - penalty

    def max_misses(self, answer):
        inv_pf = 1. / self.penalty_factor

        return int((answer / -math.log(1 + self.total_writes)) ** inv_pf)
