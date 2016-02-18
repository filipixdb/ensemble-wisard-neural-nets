import collections as cl
import itertools as it

import gmpy
import numpy as np

import bktree
import wann.wisard.discriminators as discriminators

class VGWiSARDDiscriminator(discriminators.Discriminator):
    def __init__(self):
        super(VGWiSARDDiscriminator, self).__init__(VGRAM)
        self.total_writes = 0

    def record(self, *args, **kwargs):
        super(VGWiSARDDiscriminator, self).record(*args, **kwargs)
        self.total_writes += 1

    def answer(self, observation):
        '''
        total_sum = sum(neuron.total_distance(address)
            for address, neuron in it.izip(observation, self.neurons))
        return - total_sum * 1. / len(self.neurons) / self.total_writes
        '''

        return -sum(neuron.min_distance(address)
            for address, neuron in it.izip(observation, self.neurons))

        '''
        answers = (neuron.answer(address)
            for address, neuron in it.izip(observation, self.neurons))

        return -sum((1.+a[1])*self.total_writes/np.mean(a[0]) for a in answers)
        '''

class VGRAM(object):
    def __init__(self):
        self.locations = cl.defaultdict(int)

    def __len__(self):
        return len(self.locations)

    def record(self, address):
        self.locations[address] += 1

    def answer(self, query_address):
        if query_address in self.locations:
            return [self.locations[query_address]], 0

        min_distance = float('inf')

        for address, value in self.locations.viewitems():
            distance = gmpy.popcount(address ^ query_address)
            if distance < min_distance:
                min_distance = distance
                answer_values = [value]
            elif min_distance == distance:
                answer_values.append(value)

        return answer_values, min_distance

    def min_distance(self, query_address):
        if query_address in self.locations:
            return 0

        min_distance = float('inf')

        for address in self.locations:
            min_distance = min(
                gmpy.popcount(address ^ query_address), min_distance)

            if min_distance == 1:
                break

        return min_distance

    def total_distance(self, query_address):
        return sum(gmpy.popcount(address ^ query_address) * writes
            for address, writes in self.locations.viewitems())

class VGRAM_BK:
    def __init__(self, bucket_sz):
        self.bktree = bktree.BKTree(bucket_sz)

    def record(self, observation):
        self.bktree.add(observation)

    def answer(self, observation):
        return self.bktree.nn_search(observation)

    def clone(self):
        return VGRAM_BK(self.bktree.bucket_sz)

'''
    import difflib
    class VGRAM3:
        def __init__(self):
            self.data = set()

        def record(self, observation):
            self.data.add(observation)

        def answer(self, observation):
            return len(difflib.get_close_matches(observation, self.data, 1, 0.9)) > 0
            #return min(gmpy.popcount(obs ^ observation) for obs in self.data)

        def clone(self):
            return VGRAM3()

    class VGRAM2:
        def __init__(self):
            self.data = {}

        def record(self, observation):
            observation = observation[0]
            bits_set = gmpy.popcount(observation)
            self.data.setdefault(bits_set, []).append(observation)

        def answer(self, observation):
            observation = observation[0]
            ans = float('inf')
            bits_set = gmpy.popcount(observation)
            for key in self.data:#sorted(self.data, key = lambda x: abs(x - bits_set)):
                #if abs(key - bits_set) >= ans: break
                ans = min(ans, min(gmpy.popcount(obs ^ observation) for obs in self.data[key]))
            return ans

        def clone(self):
            return VGRAM2()

    from collections import deque
    class Vgram:
        class Node:
            def __init__(self, height = 0):
                self.height = height
                self.sons = [None] * 2

        def __init__(self):
            self.nodes = [self.Node()]

        def clone(self):
            return Vgram()

        def record(self, observation, label = None):
            node = self.nodes[0]

            for bit in observation:
                if not node.sons[bit]:
                    node.sons[bit] = len(self.nodes)
                    self.nodes += [self.Node(node.height + 1)]

                node = self.nodes[node.sons[bit]]

            if hasattr(node, "sons"): del node.sons

            node.label = label

        def answer(self, observation):
            best_dist, best_groups, dq = None, {}, deque([(0, 0)])

            while dq:
                index, dist = dq.pop()
                node = self.nodes[index]

                if best_dist == None:
                    if node.height != len(observation):
                        bit = observation[node.height]

                        if node.sons[bit]:
                            dq.append((node.sons[bit], dist))

                        bit ^= 1

                        if node.sons[bit]:
                            dq.appendleft((node.sons[bit], dist + 1))
                    else:
                        best_dist = dist
                        best_groups[node.label] = 1

                elif dist == best_dist:
                    if node.height != len(observation):
                        bit = observation[node.height]

                        if node.sons[bit]:
                            dq.append((node.sons[bit], dist))
                    else:
                        best_groups[node.label] = (
                                best_groups.get(node.label, 0) + 1)
                else:
                    break

            best_count = max(best_groups.values())

            best_groups = [g[0] for g in best_groups.iteritems()
                    if g[1] == best_count]

            return best_groups, best_dist
'''
