import collections as cl
import itertools as it
import math
import random as rnd
import warnings

import Image
import numpy as np

import wann.util as util

class DataEncoder(object):
    def __init__(self):
        raise NotImplementedError('This class is abstract. Derive it.')

    def __call__(self, observation):
        raise NotImplementedError('This method is abstract. Override it.')

    def unmap(self, addresses):
        raise NotImplementedError('This method is abstract. Override it.')


class PpmEncoder(DataEncoder):
    def __init__(self, nmbr_neurons):
        self.nmbr_neurons = nmbr_neurons
        self.mapping = None

    def map(self, a_file):
        pixels = Image.open(a_file).getdata()

        if not self.mapping:
            self.mapping = range(len(pixels))
            rnd.shuffle(self.mapping)

        bits = ''.join('0' if pixels[x] <= 0 else '1'
                for x in self.mapping)
        
        return [int(bits[i::self.nmbr_neurons], 2)
                for i in xrange(self.nmbr_neurons)]


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


"""
How to calculate the reps_factor 1.0

Hypothetically:
Input: real vector, D dimensions, values with P decimal places
Output: N addresses

Each input value is represented using ceil(log(10**P, 2)) bits.
Target: map these ceil(log(10**P,2)) * D bits into N addresses.

The number of addresses an input bit is present in is
exponentially proportional to its significance in the input value:
f**i, for the i-ths bits of the input values.

A value of f = 2 seems good, as the 'importance' of the i-th bit 
is twice the one of the (i-1)-th. It also seems good that the most
significant bit is present in half of the addresses.
This way, we would need 

2 * 2 ** (ceil(log(10**P, 2)) - 1) == f ** ceil(log(10**P, 2))

neurons in our network. However, to follow the 'N addresses'
requirement, we must adjust f accordingly:

N = f ** ceil(log(10**P,2))
f = N ** (1./ceil(log(10**P,2)))
"""

"""
How to calculate the reps_factor 2.0

Hypothetically:
Input: real vector, D dimensions, values with P decimal places
Output: N addresses with B bit each, ideally

Each input value is represented using ceil(log(10**P, 2)) bits.
Target: map these ceil(log(10**P,2)) * D bits into N*B bits

The number of times a input bit should appear in the N*B output bits is
exponentially proportional to its significance in the input value: 
f**i, for the i-ths bits of the input values.

So, f must be set to generate the N*B bits:

N*B = D * sum(f**i for i in range(ceil(log(10**P,2)))) =>
sum(f**i for i in range(ceil(log(10**P,2)))) = N*B/D =>

(if f == 2, the 'ideal' case) =>

f ** (ceil(log(10 ** P, 2))) = N * B / D =>
f = (N * B / D) ** (1. / ceil(log(10 ** P, 2)))
"""

"""
'How to calculate the reps_factor 3.0'

Read first 'How to calculate the reps_factor 2.0'

Let's start from the equation

sum(f**i for i in range(ceil(log(10**P,2)))) = N*B/D .

The summation has a closed form, resulting in 

(f**ceil(log(10**P,2)) - 1)/(f - 1) = N*B/D

Considering the left side of the equation a function of f, it is
monotone, which enables the use of binary search to estimate the
value of f which solves the equation.

"""

class VectorEncoder(DataEncoder):
    def __init__(
            self, nmbr_neurons, nmbr_bits, addresses_len=None,
            len_obsrvtn=None):
        self.nmbr_neurons = nmbr_neurons
        self.nmbr_bits = nmbr_bits
        self.addresses_len = addresses_len or self.nmbr_bits
        self.mapping = self.reps = self.reps_factor = None

        if len_obsrvtn:
            self.define_mapping(len_obsrvtn)

    def __repr__(self):
        return "VectorEncoder(%d, %d, %d)" % (
            self.nmbr_neurons, self.nmbr_bits, self.addresses_len)

    def __str__(self):
        if self.mapping:
            real_addr_len = len(self.mapping) * 1. / self.nmbr_neurons
            reps_str = '%f %s %d' % (
                self.reps_factor, self.reps, sum(self.reps))
        else:
            real_addr_len = -1
            reps_str = '?'

        return "%s, real addresses length: %.2f, reps: %s" % (
            repr(self), real_addr_len, reps_str)

    #@profile
    def map(self, observation):
        if not self.mapping:
            self.define_mapping(len(observation))

        mapping_iter = iter(self.mapping)
        addresses = [0] * self.nmbr_neurons

        for val in observation:
            for i in xrange(self.nmbr_bits):
                bit = (val >> i) & 1

                for _ in it.repeat(None, self.reps[i]):
                    k = next(mapping_iter)
                    addresses[k] = (addresses[k] << 1) | bit

        return addresses

    def define_mapping(self, len_observation):
        self.calc_repetitions(len_observation)

        neurons = range(self.nmbr_neurons)
        n = self.nmbr_neurons

        self.mapping = []

        for _ in it.repeat(None, len_observation):
            for j in xrange(self.nmbr_bits):
                reps_j = self.reps[j]

                while n + reps_j > len(neurons):
                    self.mapping.extend(neurons[n:])
                    neurons[:n] = rnd.sample(neurons[:n], n)
                    reps_j -= len(neurons) - n
                    n = 0

                self.mapping.extend(neurons[n:n + reps_j])
                n += reps_j

    def calc_repetitions(self, len_observation):
        target_val = self.nmbr_neurons * self.addresses_len
        target_val /= 1. * len_observation
        func = lambda f: (f ** self.nmbr_bits - 1)/(f - 1)
        lo, hi = 0., 1.5 # initial binary search boundaries

        while func(hi) < target_val:
            hi *= 2

        for _ in it.repeat(None, 42):  # 42: enough loops and a joke
            reps_factor = (lo + hi)/2.
            if func(reps_factor) < target_val:
                lo = reps_factor
            else:
                hi = reps_factor

        self.reps_factor = reps_factor
        self.reps = [int(reps_factor ** i) for i in xrange(self.nmbr_bits)]

        warn_flag = False
        warn_msg = '\n\t%s setup apparently inadequate:' % (
            self.__class__.__name__)

        if not 1. <= reps_factor <= 2.:
            warn_flag = True
            warn_msg += '\n\t\t- reps_factor = %f.' % reps_factor

        if self.reps[-1] > self.nmbr_neurons:
            warn_flag = True
            warn_msg += (
                '\n\t\t- the number of replications of at least ' +
                'one of the input\n\t\tbits is greater than the number ' +
                'of neurons: %d > %d.') % (self.reps[-1], self.nmbr_neurons)
            
        if warn_flag:
            warn_msg += '\n\tConsider adjusting the parameters used.'
            warnings.warn(warn_msg)


class MatrixEncoder(VectorEncoder):
    def __init__(
            self, nmbr_neurons, decimal_places, addresses_len=None,
            bin_conv=lambda val: val ^ (val >> 1)):
        super(MatrixEncoder, self).__init__(
            nmbr_neurons, decimal_places, addresses_len, bin_conv)

    def map(self, obsrvtn):
        if self.mapping is None:
            self.define_mapping(len(obsrvtn))

        obsrvtn = np.array([self.bin_conv(int(i * self.pow10))
            for i in obsrvtn])

        #print obsrvtn

        #return np.dot(self.mapping, obsrvtn).astype(int).tolist()
        #return [int(i) for i in np.dot(self.mapping, obsrvtn)]
        #return [(obsrvtn * i).astype(int).sum() for i in self.mapping]
        return [np.right_shift(obsrvtn, i).sum() for i in self.mapping]

    def define_mapping(self, len_obsrvtn):
        self.calc_repetitions(len_obsrvtn)
        self.reps.append(self.nmbr_neurons)
        prev = 0
        reps_index = []
        for i, r in enumerate(self.reps):
            reps_index.extend([i] * (r - prev))
            prev += r - prev

        #print self.reps, reps_index

        self.mapping = np.array([np.rnd.permutation(reps_index)
            for i in xrange(len_obsrvtn)]).T


class MatrixEncoder2(VectorEncoder):
    def __init__(
            self, nmbr_neurons, decimal_places, addresses_len=None,
            bin_conv=lambda val: val ^ (val >> 1)):
        super(MatrixEncoder2, self).__init__(
            nmbr_neurons, decimal_places, addresses_len, bin_conv)

    def map(self, obsrvtn):
        if self.mapping is None:
            self.define_mapping(len(obsrvtn))

        obsrvtn = [self.bin_conv(int(i * self.pow10))
            for i in obsrvtn]

        return [sum(a >> b for a, b in izip(obsrvtn, i))
            for i in self.mapping]

    def define_mapping(self, len_obsrvtn):
        self.calc_repetitions(len_obsrvtn)
        self.reps.append(self.nmbr_neurons)
        prev = 0
        reps_index = []
        for i, r in enumerate(self.reps):
            reps_index.extend([i] * (r - prev))
            prev += r - prev

        mapping = (rnd.sample(reps_index, len(reps_index))
            for i in xrange(len_obsrvtn))

        self.mapping = zip(*mapping) # matrix transposal


def _type0_cell(observation, mapping, i):
    return '1' if observation[mapping[i]] < observation[mapping[i-1]] else '0'


class MinchintonEncoder(DataEncoder):
    """An encoder based on Minchinton Cells.

    Reference:
    The Minchinton Cell - Analog Input to the N-tuple Net
    P. R. Minchinton, J. M. Bishop, R. J. Mitchell
    International Neural Network Conference
    1990, p 599
    """

    def __init__(self, mul_factor=1, func=_type0_cell):
        self.mapping = None
        self.mul_factor = mul_factor
        self.func = func

    def __call__(self, observation):
        if not self.mapping:
            self.mapping = range(len(observation)) * self.mul_factor
            rnd.shuffle(self.mapping)

        return ''.join(self.func(observation, self.mapping, i)
            for i in xrange(len(self.mapping)))


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


class EncodingComposer(DataEncoder):
    """A special encoder, which just combine other encoders in a sequence.

    This should be used when to map an observation more than one encoder is
    needed. In other words, the two lines below should have the same effect:

    FooEncoder()(BarEncoder()(observation))
    EncoderSequencer(FooEncoder(), BarEncoder())(observation)
    """

    def __init__(self, *args):
        self.encoders = args[::-1]

    def __call__(self, observation):
        result = observation
        for e in self.encoders:
            result = e(result)
        return result


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
                mappings.append(collections.defaultdict(factory, m))

        self.mappings = mappings


class SortEncoder(DataEncoder):
    """An extension of the ideas present in the MinchintonEncoder.

    Instead of comparing each value to its neighbors, it uses the ranking of the
    values, which are then converted to binary.
    All values received must be in NumPy arrays.

    >>> sort_enc = SortEncoder(20)
    >>> sort_enc(np.array([3, 8, 5, 13, 12]))
    '00000011000111110111'

    It can also operate in columns instead of rows.

    >>> sort_enc_col = SortEncoder(10, use_columns=True)
    >>> sort_enc_col(np.array([[3, 8, 5, 13, 12], [10, 9, 8, 7, 6]]))
    '0101011010'
    """
    def __init__(self, total_bits, use_columns=False):
        self.total_bits = total_bits
        self.unary_encoder = None
        self.use_columns = use_columns

    def __call__(self, observation):
        if self.use_columns:
            if self.unary_encoder is None:
                rows, columns = observation.shape
                bits_per_attr = int(
                    self.total_bits * 1. / rows / columns + .5)
                self.unary_encoder = UnaryEncoder(0, rows-1, bits_per_attr)

            return ''.join(self.unary_encoder(col.argsort().argsort())
                for col in observation.T)

        if self.unary_encoder is None:
            attrs = len(observation)
            self.unary_encoder = UnaryEncoder(
                0, attrs-1, int(self.total_bits * 1. / attrs + .5))

        return self.unary_encoder(observation.argsort().argsort())
