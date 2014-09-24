#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =======
# License
# =======
# 
# pyTSA is distributed under a 3-clause ("Simplified" or "New") BSD
# license. Parts of Pandas, NumPy, SciPy, numpydoc, bottleneck, which all have
# BSD-compatible licenses, are included. Their licenses follow the pandas
# license.
# 
# pyTSA license
# =============
# 
# Copyright (c) 2014, Luca De Sano, Giulio Caravagna
# All rights reserved.
# See LICENSE.txt 

class CommentedFile(file):
    """ this class skips comment lines. comment lines start with any of the symbols in commentstring """
    def __init__(self, 
                 f, 
                 commentstring=None, 
                 low_limit=-float('inf'), 
                 high_limit=float('inf'), 
                 every=None, 
                 convert_comma=None):
    
        self.f = f
        if commentstring is None:
            self.comments = '\n'
        else:
            self.comments = commentstring + '\n'
        self.l_limit = low_limit
        self.h_limit = high_limit
        self.numrows = self.line(f)
        self.convert_comma = convert_comma
        self.readnumber = 0.0
        if every is None:
            self.every = 1.0
        elif every > 1 or every <= 0:
            raise ValueError('\n every must be in range 0.0 - 1.0')
        else:
            self.every = float(self.numrows / (self.numrows * every))

        self.rowtoadd = 0.0


    # return next line, skip lines starting with commentstring
    def next(self):
        line = self.f.next()
        while (self.readnumber != int(self.rowtoadd)):
            self.readnumber += 1
            line = self.f.next()
        self.readnumber += 1
        self.rowtoadd += self.every

        try:
            line = line.lstrip()
            while line[0] in self.comments or float(line.split()[0]) < self.l_limit:
                line = self.f.next()

            splitted = line.split()
            if  float(splitted[0]) < self.h_limit:
                if self.convert_comma:
                    return ','.join(splitted)
                return line
            else:
                self.close()
                self.readnumber = 0
                self.rowtoadd = 0
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