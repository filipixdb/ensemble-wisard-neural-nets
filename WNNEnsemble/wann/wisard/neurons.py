import collections as cl
import functools as ft

class Neuron(object):
    '''The superclass of all WiSARD-like neurons.

    This should be used as a template to any neuron implementation, as an
    abstract class, but no method is indeed required to be overriden.
    '''

    def __init__(self):
        raise NotImplementedError('This class is abstract. Derive it.')

    def __len__(self):
        '''Returns how many RAM locations are written.'''
        raise NotImplementedError('This method is abstract. Override it.')

    def __iter__(self):
        '''Returns an iterator to the wirtten RAM locations.'''
        raise NotImplementedError('This method is abstract. Override it.')

    def record(self, address):
        '''Writes the location addressed by the given argument.'''
        raise NotImplementedError('This method is abstract. Override it.')

    def answer(self, address):
        '''Returns true iff the location being addressed is written.'''
        raise NotImplementedError('This method is abstract. Override it.')

    def count(self, address):
        '''Returns how many times the location being addressed was written.'''
        raise NotImplementedError('This method is abstract. Override it.')

    def bit_counts(self):
        '''Returns how many times each bit was set in the addresses recorded.

        This method return a list 'bit_freq', where bit_freq[i] is the number
        of times the bit i was set in the addresses recorded.

        '''
        raise NotImplementedError('This method is abstract. Override it.')

    def intersection_level(self, neuron):
        '''Returns the ammount of locations written in both neurons.

        Considering a & b the intersection of the locations written in both
        neurons and a | b their union, this method returns (a & b)/(a | b).
        '''
        raise NotImplementedError('This method is abstract. Override it.')

    def bleach(self, threshold):
        '''Bleach each location written.

        The bleach operation is described as to reduce the writing count of a
        location by the given threshold if this count is over the threshold. If
        not, this location is cleaned.
        '''
        raise NotImplementedError('This method is abstract. Override it.')


class DictNeuron(Neuron):
    '''A basic neuron based on Python's dict(). PyWNN's default neuron.'''
    def __init__(self, type_=int):
        self.locations = cl.defaultdict(type_)

    def __len__(self):
        return len(self.locations)

    '''2014-05-29: Useful?
    def __iter__(self):
        return iter(self.locations)
    '''

    def record(self, address, intensity=1):
        self.locations[address] += intensity

    def answer(self, address):
        return address in self.locations

    def count(self, address):
        return self.locations.get(address, 0)

    def bit_counts(self):
        bit_freq = [0]

        for addr, freq in self.locations.viewitems():
            while addr:
                last_bit_index = (addr & -addr).bit_length() - 1
                bit_freq.extend([0] * (last_bit_index + 1 - len(bit_freq)))
                bit_freq[last_bit_index] += freq
                addr &= addr - 1  # unset last bit

        return bit_freq

    def intersection_level(self, neuron):
        # TODO: Consider probability distributions diff measures.
        # Examples: Kullback-Leibler divergence, Jensen-Shannon divergence
        len_intrsctn = len(
            self.locations.viewkeys() & neuron.locations.viewkeys())
        len_union = len(
            self.locations.viewkeys() | neuron.locations.viewkeys())
        return len_intrsctn * 1. / len_union

    def bleach(self, threshold):
        for address in self.locations.keys():
            if self.locations[address] > threshold:
                self.locations[address] -= threshold
            else:
                del self.locations[address]


class SWNeuron(DictNeuron):
    def __init__(self):
        self.locations = cl.OrderedDict()

    def record(self, address, time):
        try:
            del self.locations[address]
        except KeyError:
            pass

        self.locations[address] = time

    def answer(self, address, time_threshold):
        if address in self.locations:
            if self.locations[address] >= time_threshold:
                return True
            else:
                del self.locations[address]

        return False

    def bleach(self, threshold):
        '''Clears the locations recorded before 'threshold'.'''
        for address, time in self.locations.viewitems():
            if time < threshold:
                del self.locations[address]
            else:
                break


class MultiValueNeuron(Neuron):
    '''A neuron which stores an associative counter in each location.

    The most basic neuron virtually stores a boolean in each location, to sign
    if it is written. An evolution of this stores a counter of how many times
    it was written. A MultiValueNeuron can count how many times a location was
    written using a given key, similar to accessing the neuron as a 2D matrix:
    neuron[location][key].
    '''

    def __init__(self):
        self.locations = cl.defaultdict(ft.partial(cl.defaultdict, int))

    def __len__(self):
        return len(self.locations)

    def __iter__(self):
        return iter(self.locations)

    def record(self, address, key):
        self.locations[address][key] += 1

    def answer(self, address, key=None):
        ans = self.locations.get(address, {})

        if key is None:
            return ans.keys()

        return key in ans

    def count(self, address, key=None):
        ans = self.locations.get(address, {})

        if key is None:
            return ans

        return ans.get(key, 0)

    def remove_class(self, clss):
        for address in self.locations:
            self.locations[address].pop(clss, None)
