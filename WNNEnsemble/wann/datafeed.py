import fnmatch
import os
import sys

import Image


class DataFeed(object):
    '''The model of the classes which serve as observations sources.'''
    def __init__(self):
        raise NotImplementedError('This class is abstract. Derive it.')

    def __iter__(self):
        raise NotImplementedError('This method is abstract. Override it.')


class FilesDirsFeed(DataFeed):
    '''Traverse a folder tree from a root, providing files as observations'''
    def __init__(self, root, pattern, ordered=False):
        self.root = root
        self.pattern = pattern
        self.ordered = ordered

    def __iter__(self):
        if self.ordered:
            lst = []

        for top, _, files in os.walk(self.root):
            clss = top[len(self.root):]

            for fyle in fnmatch.filter(files, self.pattern):
                if not self.ordered:
                    yield (clss, os.path.join(top, fyle))
                else:
                    lst.append((clss, os.path.join(top, fyle)))

        if self.ordered:
            lst.sort()

            for i in lst:
                yield i


class TxtFileFeed(DataFeed):
    def __init__(self, fyle=None, sep=' ', conv=lambda x: x):
        self.fyle = fyle or sys.stdin
        self.sep = sep
        self.conv = conv

    def __iter__(self):
        for line in self.fyle:
            line = line.strip()
            if line[0] == '#':
                continue

            parts = self.conv(line.split(self.sep))
            yield parts[1] if len(parts) == 2 else parts[1:], parts[0]


class PixelsFeed(DataFeed):
    def __init__(self, imgs_data_feed, single_value=False):
        self.imgs_df = imgs_data_feed
        self.single_value = single_value

    def __iter__(self):
        for a_file in self.imgs_df:
            img = Image.open(a_file[1])
            columns = img.size[0]

            for i, pixel in enumerate(img.getdata()):
                if self.single_value:
                    pixel = pixel[0:1]

                yield (list(pixel), [i / columns, i % columns])

            yield None  # EOF marker
