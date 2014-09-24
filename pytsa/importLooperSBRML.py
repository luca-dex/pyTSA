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
import lxml.etree as etree

from multiprocessing import Process
from commentedfile import *
from Queue import Empty

class ImportLooperSBRML(Process):
    def __init__(self, 
                 path, 
                 queueIN, 
                 queueOUT, 
                 re,
                 every, 
                 tmin, 
                 tmax, 
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
        self.colnames = colnames
        self.colid = colid
        self.col_pref = col_pref
        self.convert_comma = convert_comma
        self.killReceived = False

        super(ImportLooperSBRML, self).__init__()

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

            columns = []
            time = []
            value = {}

            try:

                tree = etree.parse(actualfile)
                root = tree.getroot()

                sbrml = 'sbrml:'
                nsmap = {'sbrml': root.nsmap[None]}

                for child in root:
                    print(child)

                root.findall(sbrml + 'tupleDescription', namespaces=nsmap)

                # find time series
                # create df in toReturn
                print(columns) 
                toReturn = None

            # mmm somethings wrong here
            except ValueError:
                raise


            # maybe commentstring is worng
            except StopIteration:
                #sys.stdout.write("\b" * (progressbarlen+2))
                sys.stdout.write("\b")
                print('Warning! In file', actualfile, 'there is somethings wrong')

            self.queueOUT.put((datadictname, toReturn))

      