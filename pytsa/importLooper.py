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

import os
import sys
import pandas as pd
from multiprocessing import Process
from commentedfile import *
from Queue import Empty
import cStringIO

class ImportLooper(Process):
    def __init__(self, 
                 path, 
                 queueIN, 
                 queueOUT, 
                 re, 
                 every, 
                 tmin, 
                 tmax, 
                 commentstring,
                 delimiter, 
                 colnames, 
                 colid, 
                 col_pref, 
                 convert_comma):

        self.path = path
        self.queueIN = queueIN
        self.queueOUT = queueOUT
        self.re = re
        self.every = every
        self.tmin = tmin
        self.tmax = tmax
        self.commentstring = commentstring
        self.delimiter = delimiter
        self.colnames = colnames
        self.colid = colid
        self.col_pref = col_pref
        self.convert_comma = convert_comma
        self.killReceived = False
        super(ImportLooper, self).__init__()

    def run(self):
        while not self.killReceived:
            try:
                filename = self.queueIN.get()
            except Empty:
                break
            
            actualfile = os.path.join(self.path, filename)
            datadictname = filename
            if self.re:
                pass
                datadictname = 'f' + self.re.sub('', datadictname)

            # create a fake file and pd.read_csv!
            toReturn = None
            try:
                source = CommentedFile(open(actualfile, 'rb'), every=self.every, \
                    commentstring=self.commentstring, low_limit=self.tmin, high_limit=self.tmax, \
                    convert_comma=self.convert_comma)
                if self.convert_comma:
                    temp_string = ""
                    for r in source:
                        temp_string = temp_string + r + '\n'
                    toReturn = pd.read_csv(cStringIO.StringIO(temp_string), sep=self.delimiter, index_col=0, \
                        header=None, names=self.colnames, usecols=self.colid, prefix=self.col_pref)
                else:
                    toReturn = pd.read_csv(source, sep=self.delimiter, index_col=0, \
                        header=None, names=self.colnames, usecols=self.colid, prefix=self.col_pref)
                source.close()

            # mmm somethings wrong here
            except ValueError:
                raise


            # maybe commentstring is worng
            except StopIteration:
                #sys.stdout.write("\b" * (progressbarlen+2))
                sys.stdout.write("\b")
                print('Warning! In file', actualfile, 'a line starts with NaN')

            if toReturn:
                self.queueOUT.put((datadictname, toReturn))
            else:
                print("Error in file", actualfile,"\n")


            