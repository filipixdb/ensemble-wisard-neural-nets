'''
@author: filipi
'''

# Metodos para fazer encoding das features

'''
BitStringEncoder
  recebe cadeia de bits, separa e organiza numa lista pra wisard
UnaryEncoder
  faz o encode unario pra um atributo numerico
  recebe um array dos valores e retorna encodado e concatenado
EncodingAppender
  faz uma lista de encodes que processam uma lista de valores respectivamente
QualitativeEncoder
  faz o encode equidistante para atributos qualitativos nao ordinais
'''

import wann.util as util

import random as rnd
import itertools as it
import numpy as np
import math
import collections as cl




class DataEncoder(object):
    def __init__(self):
        raise NotImplementedError('This class is abstract. Derive it.')

    def __call__(self, observation):
        raise NotImplementedError('This method is abstract. Override it.')

    def unmap(self, addresses):
        raise NotImplementedError('This method is abstract. Override it.')



class BitStringEncoder(DataEncoder):
    """A bit string to addresses (integers) list encoder.

    Based on a mapping randomly defined on the first object call,
    encodes the bits  of a given bit string into a list of addresses
    suitable to be input 0to a WiSARD discriminator.

    Attributes:
        nmbr_neurons: the length of the addresses list to be generated.
        addr_len: the maximum number of bits of each address.
        mapping: the mapping used to define the addresses.
        reverse_mapping: a mapping which could be used to retrieve the bit
            string used to produce an addresses list.
    """

    def __init__(self, nmbr_neurons=None, addr_len=None):
        if nmbr_neurons is None and addr_len is None:
            raise TypeError(
                'if nmbr_neurons is None, then addr_len must not be None.')
        self.addr_len = addr_len
        self.nmbr_neurons = nmbr_neurons
        self.mapping = self.reverse_mapping = None

    def __call__(self, bit_string):
        """Encodes a bit string, according to a mapping.

        The mapping is randomly defined on the first call to this method.

        Returns:
            BitStringEncoder(2)('11110001') would return, considering that
            mapping = [7, 6, 1, 5, 2, 4, 3, 0]:
                '11110001' ======== mapping ===========> '10101011',
                slices are broken using the number of neurons as step, so:
                '10101011' ======== splitting =========> ['1111', '0001'],
                ['1111', '0001'] == making addresses ==> [15, 1].

            BitStringEncoder(2)('11110001') = [15, 1]
        """

        if not self.mapping:
            if self.nmbr_neurons is None:
                self.nmbr_neurons = int(
                    round(len(bit_string) * 1. / self.addr_len))

            self.mapping = []

            """
            The mapping is defined by shuffled ranges of length
            self.nmbr_neurons. The bits that will compose the n-th
            address are indexed by the n-th value of each of these
            ranges. This by-range shuffling targets avoiding close,
            possibly correlated bits to compose the same address.
            """

            for i in xrange(0, len(bit_string), self.nmbr_neurons):
                ax = range(i, min(i + self.nmbr_neurons, len(bit_string)))
                rnd.shuffle(ax)
                self.mapping.extend(ax)

        bits = ''.join([bit_string[x] for x in self.mapping])

        return [int(bits[i::self.nmbr_neurons], 2)
            for i in xrange(self.nmbr_neurons)]
    
    
    def select_features(self):
        '''
        Precisa receber:
          array de features a usar
          array com o tamanho de cada feature, no caso dummie todas tem 8 bits
          talvez de pra fazer soh com o mapping
          vai ter que parar de usar em alguns calculos o tamanho todo das instancias
          
          
        '''
    
    
    

    #TODO: rewrite the unmap method
    """
    def unmap(self, addresses):
        if self.nmbr_neurons == 1:
            s = bin(addresses)[2:]
            return '0' * (len(self.mapping) - len(s)) + s

        min_addr_len = len(self.mapping) / self.nmbr_neurons
        plus_one_lim = len(self.mapping) % self.nmbr_neurons

        def plus_one_fix(a, i):
            if len(a) == min_addr_len + plus_one_lim:
                return a

            return '0' + a if i < plus_one_lim else a + '0'

        bits = ''.join(
                plus_one_fix(a.zfill(min_addr_len), i)
                     for i, a in enumerate(bin(a)[2:] for a in addresses))

        if plus_one_lim: min_addr_len += 1
        bits = ''.join(bits[i::min_addr_len] for i in xrange(min_addr_len))

        if not self.reverse_mapping:
            self.reverse_mapping = [0] * len(self.mapping)

            for i, j in enumerate(self.mapping):
                self.reverse_mapping[j] = i

        return ''.join(bits[x] for x in self.reverse_mapping)
    """

class UnaryEncoder(DataEncoder):
    """A numeric observation encoder based on unary representations.

    In the unary numeral system a number is represented by as many ones as the
    value of the number itself:
        3 -> 111
        6 -> 111111
        ...

    Let the maximum value (mv) to deal with be known in advance. This way,
    the length of the unary representation of a number is at most mv. All
    representations could have the same length by adding zeros to the left of
    those of numbers less than mv. For example, if mv = 8:
        unary(8) = '11111111', unary(3) = '00000111'

    To use a smaller number of bits (nb) to represent the values, a simple rule
    of three can be used. For example, if mv = 8 and nb = 4:
        unary(8) => unary(nb * 8 / mv) => unary(4) = '1111'
        unary(3) => unary(nb * 3 / mv) => unary(1) = '0001'

    Attributes:
        max_value: the maximum value to be considered to define the unary
            representation of the values of an observation. It is possible to
            specify the maximum value to be used for each dimension by
            providing a list through this parameter, instead of just a number.

        min_value: the minimum value to be considered. This has the same
            properties of max_value.

        number_bits: the number of bits to be used to represent the numeric
            values. It is possible to specify the number of bits to be used for
            each dimension by providing a list through this parameter, instead
            of just a number.

    Note:
        This encoder DO NOT produces addresses lists! Another encoder, as the
        BitStringEncoder, should be used to obtain these.
    """

    def __init__(self, min_or_max, max_value=None, number_bits=None, cyclic=False):
        """Inits UnaryEncoder using the provided arguments.
        
        If min_value is not provided, it is equaled to 0. If number_bits is
        not provided, it is equaled to max_value - min_value.
        """

        min_value = min_or_max
        if max_value is None:
            max_value = min_or_max
            min_value = 0

        self.__setstate__([min_value, max_value, number_bits, cyclic])

    def __call__(self, obsrvtn):
        """Maps an observation to its unary representation.

        Returns:
            A concatenetion of the unary representations of the values of a
            given observation. For example, UnaryEncoder(0, 8, 4)([8,3])
            returns '11110011'.

        Args:
            obsrvtn: the numeric array to be mapped
        """

        mapped = ''

        for v, minv, maxv, nb, cyclic in it.izip(obsrvtn, *self.parameters):
            if nb is None:
                nb = maxv - minv

            v = 1. * (util.clip(v, minv, maxv) - minv)
            v /= (maxv - minv + np.finfo(float).eps)

            if cyclic:
                v *= 2*math.pi
                v = [(f(v) + 1)/2 for f in (math.sin, math.cos)]
                s = (('1' * int(round(v[0] * nb/2))).zfill(nb/2) +
                     ('1' * int(round(v[1] * nb/2))).zfill(nb/2))
                mapped += s.zfill(nb)
            else:
                mapped += ('1' * int(v*nb + .5)).zfill(nb)

        return mapped

    def __getstate__(self):
        return [next(p) if isinstance(p, repeat) else p
            for p in self.parameters]

    def __setstate__(self, parameters):
        self.parameters = []

        for parameter in parameters:
            try:
                iter(parameter)
                self.parameters.append(parameter)
            except TypeError:
                self.parameters.append(it.repeat(parameter))


class EncodingAppender(DataEncoder):
    """A special encoder to apply a sequence of encoders to a sequence of
    attributs.

    This should be used when to map a sequence of observations that need to be
    treated by different encoders. The following lines work as an example:

    >>> appender = EncodingAppender(quali, unary)
    >>> appender(['a', 5]) == [quali(['a']), unary([5])]
    True
    """
    def __init__(self, *args):
        self.encoders = args

    def __call__(self, observation):
        result = []

        for enc, val in zip(self.encoders, observation):
            result.append(enc([val]))

        return result

    def add_encoder(self, encoder):
        self.encoders.append(encoder)


# TODO: Test another criterium: maximal minimal distance and maximal sum of distances - DAlves
class QualitativeEncoder(DataEncoder):
    """An enconder for qualitative data.

    This enconder associates binary strings to categorical data,
    taking into consideration that a collection of values of a
    categorical (qualitative) variable are equidistant from each other,
    and this should be reflected in their binary representations.

    This encoder uses util.disjunct_bitstrings function. Its
    documentation should be checked in case of problems in
    QualitativeEncoder use.
    """
    def __init__(self, number_bits, repeat_bits=None):
        self.number_bits = number_bits
        self.repeat_bits = repeat_bits
        self.mappings = []

    def __call__(self, observation):
        mapped = ''

        for i, v in enumerate(observation):
            try:
                mapped += self.mappings[i][v]

            except IndexError:
                try:
                    nb = self.number_bits[i]
                except (IndexError, TypeError):
                    nb = self.number_bits

                try:
                    rb = self.repeat_bits[i]
                except (IndexError, TypeError):
                    rb = self.repeat_bits

                factory = util.iterator2factory(
                    util.disjunct_bitstrings(nb, rb))
                self.mappings.append(cl.defaultdict(factory))

                mapped += self.mappings[i][v]

        return mapped

    def __getstate__(self):
        dict_ = self.__dict__.copy()
        dict_['mappings'] = [m.items() for m in dict_['mappings']]
        return dict_

    def __setstate__(self, dict_):
        self.__dict__.update(dict_)

        mappings = []
        for i, m in enumerate(self.mappings):
            try:
                nb = self.number_bits[i]
            except (IndexError, TypeError):
                nb = self.number_bits

            try:
                rb = self.repeat_bits[i]
            except (IndexError, TypeError):
                rb = self.repeat_bits

                factory = util.iterator2factory(
                    util.disjunct_bitstrings(nb, rb))
                mappings.append(cl.defaultdict(factory, m))

        self.mappings = mappings

