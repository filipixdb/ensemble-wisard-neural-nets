import gmpy

class BKTree:
    def __init__(self, bucket_sz):
        self.bucket_sz = bucket_sz
        self.root = BKNode(self)
        #self.xx = 0L

    def add(self, key):
        self.root.add(key)

    def nn_search(self, key):
        self.key = key
        self.min_dist = float('inf')
        self.root.nn_search()
        return self.min_dist

class BKNode:
    def __init__(self, bktree):
        self.bktree = bktree
        self.pivot = None
        self.bucket = []

    def add(self, key):
        if self.pivot == None:
            self.bucket.append(key)

            if len(self.bucket) > self.bktree.bucket_sz:
                self.pivot = self.bucket.pop()

                self.childs = {}
                for k in self.bucket:
                    self.childs.setdefault(
                            gmpy.popcount(self.pivot ^ k), 
                            BKNode(self.bktree)).add(k)

                del self.bucket
        else:
            self.childs.setdefault(
                    gmpy.popcount(self.pivot ^ key), 
                    BKNode(self.bktree)).add(key)

    def nn_search(self):
        if self.pivot == None:
            #self.bktree.xx += len(self.bucket)
            dist = min(gmpy.popcount(self.bktree.key ^ k)
                    for k in self.bucket)

            self.bktree.min_dist = min(
                    self.bktree.min_dist,
                    dist)
        else:
            #self.bktree.xx += 1
            dist_root = gmpy.popcount(
                    self.bktree.key ^ self.pivot)

            self.bktree.min_dist = min(
                    self.bktree.min_dist,
                    dist_root)
            
            for dist in sorted(
                    self.childs,
                    key = lambda x: abs(x - dist_root)):
                if abs(dist_root - dist) > self.bktree.min_dist:
                    break

                self.childs[dist].nn_search()

'''
import random as r
import timeit as t
a = [r.randint(0,2**10) for i in xrange(2**10)]
b = [r.randint(0,2**10) for i in xrange(2**10)]

bkt = BKTree(34)

for i in b:
    bkt.add(i)

for c in a:
    if not min(gmpy.popcount(c ^ d) for d in b) == bkt.nn_search(c):
        print 'bad'

print bkt.xx, len(a) * len(b)

print t.timeit(
    'for c in a: min(gmpy.popcount(c ^ d) for d in b)',
    'from __main__ import a, b, bkt, gmpy',
    number = 1)

print t.timeit(
    'for c in a: bkt.nn_search(c)',
    'from __main__ import a, b, bkt, gmpy',
    number = 1)
'''
