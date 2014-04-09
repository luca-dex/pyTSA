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

from Queue import Empty
from multiprocessing import Process

class DataSampler(Process):
    def __init__(self, 
                 queueIN, 
                 queueOUT, 
                 tmin, 
                 tmax, 
                 step, 
                 colname):
    
        self.queueIN = queueIN
        self.queueOUT = queueOUT
        self.tmin = tmin
        self.tmax = tmax
        self.step = step
        self.colname = colname
        self.killReceived = False
        super(DataSampler, self).__init__()

    def run(self):
        while not self.killReceived:
            try:
                k,v = self.queueIN.get()
            except Empty:
                break
            sampled = self.get(v[self.colname], self.tmin, self.tmax, self.step)
            self.queueOUT.put((k, sampled))

    @staticmethod
    def get(df, l_limit, h_limit, step):
        start = float(l_limit)
        now = float(start + step)
        to_return = []
        try:
            last_value = df.truncate(after=start).tail(1).values[0]
        except:
            last_value = 0
        to_return.append(last_value)
        while now < h_limit:
            try:
                last_value = df.truncate(before=start, after=now).tail(1).values[0]
            except:
                last_value = to_return[-1]
            to_return.append(last_value)
            start = start + step
            now = now + step
        return to_return
