import collections as cl
import functools as ft
import heapq
import itertools as it

import numpy as np


def _answers_sortkey(item):
    try:
        return len(item[1]), item[1]
    except TypeError:
        return item[1]


def ranked(dict_, key_func=_answers_sortkey):
    """Returns a list of the items in dict_ sorted by key_func."""
    return sorted(dict_.viewitems(), key=key_func, reverse=1)


def iter_llist_items(llist):
    """Yields a tuple (node, node.value) for each node in llist"""
    node = llist.first
    while node:
        yield node, node.value
        node = node.next


class ClusterInfo:
    """The container of all information of a cluster."""
    def __init__(self):
        self.absorptions = 0
        self.class_ = None

    def __repr__(self):
        return str((self.absorptions, self.class_))


def disjunct_bitstrings(total_bits, repeat_bits=None):
    """Yields bit strings as Hamming-disjunct as possible.

    Yields bit strings of length total_bits which are as far as
    possible (wrt Hamming distance) from previously yielded strings.

    If repeat_bits is None, all 2**total_bits strings between 0 and
    2**total_bits - 1 will be yielded in some order. Else,
    2**repeat_bits strings will be yielded.

    The strings are ordered in a brute-force fashion. So, the use this
    function for arbitary values is time-prohibitive: total_bits, or
    repeat_bits when in use, should be smaller than 64.
    """
    if repeat_bits is not None:
        repetitions = int(np.ceil(total_bits * 1. / repeat_bits))

        for mask in disjunct_bitstrings(repeat_bits):
            yield (mask*repetitions)[:total_bits]

        return

    #TODO: make a better, generic, cross-plataform check for overflow
    if total_bits > 63:
        raise OverflowError(
            'Please lower the requested number of bits: %d' % total_bits)

    yield '0'*total_bits
    yield '1'*total_bits

    olds = [0, 2**total_bits - 1]

    for distance in xrange(total_bits/2, 0, -1):
        for value in xrange(1, 2**total_bits-1):
            min_distance = min(bin(value ^ old).count('1') for old in olds)

            if min_distance == distance:
                olds.append(value)
                yield bin(value)[2:].zfill(total_bits)


def iterator2factory(iterator):
    """Returns a function which returns next(iterator) at each call."""
    return lambda: next(iterator)


def iterative_bleaching(counts, big_steps=False):
    #TODO: Complete review
    bleach_level = 0

    while 1:
        if big_steps:
            try:
                min_count = max(counts.viewitems(), key=_answers_sortkey)[1][0]
            except IndexError:
                return

        else:
            try:
                min_count = min(v[0] for v in counts.values() if v)
            except ValueError:
                return

        yield counts, bleach_level, min_count

        bleach_level += min_count

        for class_, class_counts in counts.viewitems():
            counts[class_] = [
                cc-min_count for cc in class_counts if cc > min_count]


class ConfusionMatrix:
    """A summary structure of the results of a classification run."""
    def __init__(self, dec_places=2):
        self.dec_places = dec_places
        self.mat = cl.defaultdict(ft.partial(cl.defaultdict, list))
        self.tc_counts = cl.defaultdict(int)
        self.sc_counts = cl.defaultdict(int)

    def add(self, true_class, said_class, answer_level=0):
        """Adds an classification outcome to the summary data."""
        self.mat[true_class][said_class].append(answer_level)
        self.tc_counts[true_class] += 1
        self.sc_counts[said_class] += 1

    def __str__(self):
        keys = sorted(self.mat.keys())
        table = []
        header = ['@', '@', '@', ' ' * (3 + self.dec_places)]
        footer = ['@', '@', '@', ' ' * (3 + self.dec_places)]
        
        sep_length = 0

        eps = np.finfo(float).eps

        for k in keys:
            ax = '@%s;' % k
            bx = '@%d;' % self.sc_counts[k]
            sep_length = max(sep_length, len(ax), len(bx))
            header.extend((ax, ' ' * (3 + self.dec_places), '@', '@'))
            footer.extend((bx, ' ' * (3 + self.dec_places), '@', '@'))
            table.extend(('@%s;' % k, '@'))

            for l in keys:
                cnt = len(self.mat[k][l])
                row_portion = cnt * 1. / (self.tc_counts[k] + eps)
                col_portion = cnt * 1. / (self.sc_counts[l] + eps)
                mean_al = np.mean(self.mat[k][l]) if self.mat[k][l] else 0.
                ax = '%.*f,' % (self.dec_places, row_portion)
                bx = '@%.*f,' % (self.dec_places, col_portion)
                cx = '@%.*f;' % (self.dec_places, mean_al)
                sep_length = max(sep_length, len(ax), len(bx), len(cx))
                table.extend((ax, bx, cx, '@'))

            table.append('%d\n' % self.tc_counts[k])

        header = ['%*s' % (sep_length, s[1:]) if s[0] == '@' else s
            for s in header]
        footer = ['%*s' % (sep_length, s[1:]) if s[0] == '@' else s
            for s in footer]
        table = ['%*s' % (sep_length, s[1:]) if s[0] == '@' else s
            for s in table]

        header.append('\n')

        return ''.join(''.join(ax) for ax in (header, table, footer))

    def stats(self, tipo=None, custos=None):
        """Return a list of popular classification quality measures."""
        total = sum(self.tc_counts.values())
        hits = 0
        precision, recall, f1_score = [], [], []
        eps = np.finfo(float).eps

        for k in self.mat:
            tp = len(self.mat[k][k])
            fp = self.sc_counts[k] - tp
            fn = self.tc_counts[k] - tp
            ratio = self.tc_counts[k] * 1. / total

            precision.append(ratio * tp * 1. / (tp + fp + eps))
            recall.append(ratio * tp * 1. / (tp + fn))
            f1_score.append(ratio * 2. * tp / (2*tp + fp + fn))

            hits += tp

        
        # calcular as penalidades do problema
        tuplaCusto = None
        if custos != None:
            penalizado = 0.0
            totalCusto = 0.0
            for classe, custo in zip(self.mat, custos):
                tp = len(self.mat[classe][classe])
                fp = self.sc_counts[classe] - tp
                fn = self.tc_counts[classe] - tp
                penalizado += fn*custo
                totalCusto += self.tc_counts[classe]*custo
            tuplaCusto = ('custo', "{0:.3f}".format(penalizado/totalCusto))
            

        if tipo == 'simples':
            return [
                ('accuracy', "{0:.3f}".format(hits * 1. / total)),
                ('avg_f1_score', "{0:.3f}".format(sum(f1_score))),
                tuplaCusto
                ]

        return [
            ('total', total),
            ('hits', hits),
            ('accuracy', hits * 1. / total),
            ('avg_precision', sum(precision)),
            ('avg_recall', sum(recall)),
            ('avg_f1_score', sum(f1_score)),
            tuplaCusto
            ]


def relevance(data, nbins=100):
    class_lines = cl.defaultdict(list)
    for i, c in enumerate(data.T[-1]):
        class_lines[c].append(i)

    ncols = data.shape[1] - 1
        
    relevance = [[] for _ in range(ncols)]
    all_hists = []

    for i, column in enumerate(data.T[:-1]):
        range_ = column.min(), column.max()
        hists = [np.histogram(column[lines], nbins, range_)[0]
            for lines in class_lines.viewvalues()]

        hists = [h * 1. / h.sum() for h in hists]

        all_hists.append(hists)

        for z in zip(*hists):
            z = heapq.nlargest(2, z)
            relevance[i].append(z[0] - z[1])

    relevance = np.array([np.mean(r) for r in relevance])

    return relevance/relevance.sum()

    redundancy  = np.zeros((ncols,ncols))

    for i, j in it.combinations(range(ncols), 2):
        ax = 0
        for lines in class_lines.viewvalues():
            h = np.histogram2d(data.T[i][lines], data.T[j][lines], nbins)[0]
            ax += (h.max(0) * 1. / (h.sum(0)+np.finfo(float).eps)).mean()
            ax += (h.max(1) * 1. / (h.sum(1)+np.finfo(float).eps)).mean()

            '''
            h = h.ravel() * 1. / h.sum()

            h.sort()
            ax += (h[1:] - h[:-1]).sum()
            h = heapq.nlargest(2, h) 
            ax += h[0]-h[1]
            '''

        redundancy[i][j] = redundancy[j][i] = ax/(2. * len(class_lines))

    return relevance/relevance.sum(), redundancy


def clip(v, min_, max_):
    return min_ if v < min_ else max_ if v > max_ else v
