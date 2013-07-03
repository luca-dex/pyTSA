# -*- coding: utf-8 -*-

import os
# import pandas as pd
# import numpy as np
# import scipy.stats as stats
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d

class CommentedFile(file):
    """ this class skips comment lines. comment lines start with any of the symbols in commentstring """
    def __init__(self, f, commentstring=None, low_limit=-float('inf'), high_limit=float('inf'), every=None):
        self.f = f
        if commentstring is None:
            self.commentstring = ''
        else:
            self.commentstring = commentstring
        self.l_limit = low_limit
        self.h_limit = high_limit
        self.numrows = self.line(f)
        self.readnumber = 0
        if every is None:
            self.every = int(self.numrows / 10)
        else:
            self.every = every


    # return next line, skip lines starting with commentstring
    def next(self):
        line = self.f.next()
        while ((self.readnumber % self.every) != 0):
            self.readnumber += 1
            line = self.f.next()
        self.readnumber += 1
        comments = self.commentstring + '\n'
        try:
            while line[0] in comments or float(line.split()[0]) < self.l_limit:
                line = self.f.next()

            if  float(line.split()[0]) < self.h_limit:
                return line
            else:
                self.close()
                self.readnumber = 0
                raise StopIteration

        except ValueError:
            self.close()
            raise StopIteration
    
    # moves the cursor to the initial position
    def seek(self):
        self.f.seek(0)

    def close(self):
        self.f.close()

    @staticmethod
    def line(f):
        numlines = 0
        for _ in f:
            numlines += 1
        f.seek(0)
        return numlines


    def __iter__(self):
        return self