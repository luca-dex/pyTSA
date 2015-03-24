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


def printFromSeries(name, title, data):
        filename = name + '.data'
        with open(filename, 'w') as f:
            f.write(title)
            f.write('\n')
            for row in zip(*data):
                cols = [c for c in row]
                for col in cols:
                    f.write(str(col))
                    f.write('\t')
                f.write('\n')

def columnsCheck(columns, col):
        if col is None:
            return columns
        if isinstance(col, str):
            col = col.split()
        for c in col:
            if c not in columns:
                error = 'Column ' + c + ' not in columns'
                raise ValueError(error)
        return col

def columnsPhSpCheck(columns, col):
        if isinstance(col, list):
            if isinstance(col[0], list):
                for c in col:
                    if len(c) != 2 or c[0] not in columns or c[1] not in columns:
                        raise ValueError('There is a problem on selected columns')
            else:
                if len(col) != 2 or col[0] not in columns or col[1] not in columns:
                    raise ValueError('There is a problem on selected columns')
                return [col]
        return col 

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
