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
                 col_pref):

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
            index = []
            indexName = ''
            value = []

            try:

                tree = etree.parse(actualfile)
                root = tree.getroot()

                sbrml = 'sbrml:'
                nsmap = {'sbrml': root.nsmap[None]}

                operations = root.find(sbrml+'operations', namespaces=nsmap)

                # find time series

                for operation in operations:
                    result = operation.find(sbrml+'result', namespaces=nsmap)
                    resultComponent = result.find(sbrml+'resultComponent', namespaces=nsmap)
                    dimDesc = resultComponent.find(sbrml+'dimensionDescription', namespaces=nsmap)
                    compDesc = dimDesc.find(sbrml+'compositeDescription', namespaces=nsmap)
                    indexName = compDesc.get('name')
                    tupleDesc = compDesc.find(sbrml+'tupleDescription', namespaces=nsmap)
                    for atomicDesc in tupleDesc:
                        columns.append(atomicDesc.get('name'))
                        value.append([])

                    dim = resultComponent.find(sbrml+'dimension', namespaces=nsmap)
                    for compValue in dim:
                        index.append(float(compValue.get('indexValue')))
                        tupleValues = compValue.find(sbrml+'tuple', namespaces=nsmap)
                        for k, atomicValue in enumerate(tupleValues):
                            value[k].append(float(atomicValue.text))

                # create a Pandas Series for every column
                s = {}
                for k, col in enumerate(columns):
                    s[col] = pd.Series(value[k], index=index)

                # merge Series in a DataFrame
                df = pd.DataFrame(s)
                df.index.name = indexName

                # put df in toReturn
                toReturn = df

            # mmm somethings wrong here
            except ValueError:
                raise


            # maybe commentstring is worng
            except StopIteration:
                #sys.stdout.write("\b" * (progressbarlen+2))
                sys.stdout.write("\b")
                print('Warning! In file', actualfile, 'there is somethings wrong')

            self.queueOUT.put((datadictname, toReturn))

      