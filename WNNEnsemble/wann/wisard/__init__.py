import collections

import llist

import classifiers
import discriminators
import neurons


class StreamWiSARD(object):
    def __init__(
            self, horizon, min_similarity,
            expected_absorptions, discriminator=None):
        self.horizon = horizon
        self.min_similarity = min_similarity
        self.expected_absorptions = expected_absorptions

        # default: SWDiscriminator(len(observation))
        self.discriminator = discriminator

        self.clear()

    def clear(self):
        self.discriminators = llist.dllist()
        # defaultdict faster than Counter (Python 2.7.3, 2013/02/22)
        self.absorptions = collections.defaultdict(int)

        self.last_absorption = {
            'accepted_answer': None, 'highest_answer': None,
            'accepted_dscrmntr_id': None}

    def absorb(self, observation, time):
        if not self.discriminator:  # set it to the default
            self.discriminator = discriminators.SWDiscriminator(
                len(observation))

        highest_answer = 0.

        list_node = self.discriminators.first
        while list_node:
            idx, d = list_node.value

            answer = d.answer(observation, time - self.horizon, True)

            if not d:
                del self.absorptions[idx]
                # trick to del the current node while advancing to the next
                list_node, spam = (
                    list_node.next, self.discriminators.remove(list_node))
                continue

            highest_answer = max(highest_answer, answer)
            ax = self.absorptions[idx] * 1. / self.expected_absorptions

            if answer >= min(ax + self.min_similarity, 1.):
                self.discriminators.remove(list_node)
                accepted_answer = answer
                break

            list_node = list_node.next

        else:
            idx, d = len(self.discriminators), self.discriminator.clone()
            d.creation_time = time
            accepted_answer = None

        d.record(observation, time)
        self.discriminators.appendleft((idx, d))
        self.absorptions[idx] += 1

        self.last_absorption.update({
            'highest_answer': highest_answer,
            'accepted_answer': accepted_answer,
            'accepted_dscrmntr_id': idx})


'''
mcd = MultiClassDiscriminator(4).clone()
mcd.record([1,2,3,4], 0)
mcd.record([1,2,0,0], 0)
mcd.record([0,0,0,0], 1)
print mcd.answer([0,2,0,9], 1)
print mcd.counts([0,2,0,9])
'''


'''
class StreamWiSARD2(object):
    def __init__(
            self, horizon, min_similarity,
            expected_absorptions, discriminator = None):
        self.horizon = horizon
        self.min_similarity = min_similarity
        self.expected_absorptions = expected_absorptions

        # default: MultiClassDiscriminator(len(observation))
        self.discriminator = discriminator

        self.clear()

    def clear(self):
        self.discriminators = llist.dllist()
        # defaultdict faster than Counter (Python 2.7.3, 2013/02/22)
        self.absorptions = collections.defaultdict(int)

        self.last_absorption = {
            'accepted_answer': None, 'highest_answer': None,
            'accepted_dscrmntr_id': None}

    def absorb(self, observation, time):
        if not self.discriminator: # set it to the default
            self.discriminator = MultiClassDiscriminator(len(observation))

        answer = self.discriminator.answer(observation, True)

        if answer >= min(ax + self.min_similarity, 1.):
            self.discriminators.remove(list_node)
            accepted_answer = answer
            break

        highest_answer = 0.

        list_node = self.discriminators.first
        while list_node:
            idx, d = list_node.value

            #d.dump_old_data(time - self.horizon)
            if not d:
                del self.absorptions[idx]
                # trick to del the current node while advancing to the next
                list_node, spam = (
                    list_node.next, self.discriminators.remove(list_node))
                continue

            answer = d.answer(observation, True)
            highest_answer = max(highest_answer, answer)
            ax = self.absorptions[idx] * 1. / self.expected_absorptions

            if answer >= min(ax + self.min_similarity, 1.):
                self.discriminators.remove(list_node)
                accepted_answer = answer
                break

            list_node = list_node.next

        else:
            idx, d = len(self.discriminators), self.discriminator.clone()
            d.creation_time = time
            accepted_answer = None

        d.record(observation, time)
        self.discriminators.appendleft((idx, d))
        self.absorptions[idx] += 1

        self.last_absorption.update({
            'highest_answer': highest_answer,
            'accepted_answer': accepted_answer,
            'accepted_dscrmntr_id': idx})
'''
