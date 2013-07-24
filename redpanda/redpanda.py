# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import axes3d
from commentedfile import *

def dataset(path, commentstring=None, colnames=None, delimiter='[\s\t]+', start=-float('inf'), stop=float('inf'), \
    colid=None, ext=None, every=None):
    '''more than one file'''
    
    #microvalidation
    if start > stop:
        print 'maybe start > stop ? \n'
    if colnames and colid:
        if len(colnames) != len(colid):
            print 'colid and colnames must have same length!'
    if colnames is None:
        col_pref = 'Y'
    else:
        col_pref = None

    # if ext and not ext.startswith('.'):
    #    ext = ''.join(('.', ext))

    if colid and delimiter != ',':
        print 'column selection work only with delimiter = \',\' (yet)'


    dataset = {}
    numberoffile = len([f for f in os.listdir(path) if os.path.isfile(path + f)])
    progressbarlen = 50
    atraitevery = int(numberoffile / progressbarlen) + 1
    counter = 0
    if (numberoffile % progressbarlen) != 0:
        progressbarlen -= 1

    sys.stdout.write("[%s]" % (" " * progressbarlen))
    sys.stdout.flush()
    sys.stdout.write("\b" * (progressbarlen+1)) # return to start of line, after '['

    # skip dir, parse all file matching ext
    for filename in os.listdir(path):
        actualfile = os.path.join(path, filename)

        # check if isdir
        if os.path.isdir(actualfile):
            continue

        # check if ext match
        if ext and not filename.endswith(ext):
            continue

        # import
        try:
            source = CommentedFile(open(actualfile, 'rb'), every=every, \
                commentstring=commentstring, low_limit=start, high_limit=stop)
            dataset[filename] = pd.read_csv(source, sep=delimiter, index_col=0, \
                header=None, names=colnames, usecols=colid, prefix=col_pref)
            source.close()

        except ValueError:
            raise
            break

        except StopIteration:
            sys.stdout.write("\b" * (progressbarlen+2))
            print 'Warning! In file %s a line starts with NaN' % actualfile
            break

        if (counter % atraitevery) == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

        counter += 1

    sys.stdout.write("\n")

    # return RedPanda Obj (isset = True)
    return RedPanda(dataset, True)

def timeseries(path, commentstring=None, colnames=None, delimiter='[\s\t]+', start=-float('inf'), stop=float('inf'), \
    colid=None, every=None):
    '''just one file'''
    
    # microvalidation
    if start > stop:
        print 'maybe start > stop ? \n'
    if colnames and colid:
        if len(colnames) != len(colid):
            print 'colid and colnames must have same length!'
    if not colnames:
        col_pref = 'Y'
    else:
        col_pref = None

    if colid and delimiter != ',':
        print 'column selection work only with delimiter = \',\' (yet)' 

    source = CommentedFile(open(path, 'rb'), \
        commentstring=commentstring, low_limit=start, high_limit=stop, every=every)
    timeseries = pd.read_csv(source, sep=delimiter, index_col=0, \
        header=None, names=colnames, usecols=colid, prefix=col_pref)
    source.close()

    # return RedPanda Obj (isset = False)
    return RedPanda(timeseries, None)

class RedPanda:
    def __init__(self, data, isSet):
        # dataset or timeseries
        self.data = data
        # what's the type
        self.isSet = isSet
        # start with default terminal (supported type are png, pdf, ps, eps and svg)
        self.outputs = set()
        self.outputs.add('view')
        # range -> label:data (pandas df with index)
        self.range = {}
        self.row = {}

    def createrange(self, label, colname, start, stop, step):
        """Select 1 column and create a range from start to stop"""
        if not self.isSet:
            print 'createrange works only on dataset'
            return
        index = np.arange(start, stop, step)
        mean_df = pd.DataFrame(index=index)
        for k,v in self.data.iteritems():
            mean_df.insert(0, k, self.get(v[colname], start, stop, step))
        self.range[label] = mean_df

    def addoutput(self, out):
        """select outputs from png, pdf, ps, eps and svg"""
        if out in ['png', 'pdf', 'ps', 'eps', 'svg', 'view']:
            self.outputs.add(out)
        else:
            print '%s not in outputs' % out

    def deloutput(self, out):
        """select outputs from png, pdf, ps, eps and svg"""
        if out in ['png', 'pdf', 'ps', 'eps', 'svg', 'view']:
            try:
                self.outputs.remove(out)
            except:
                print '%s not in outputs' % out
        else:
            print '%s not in outputs' % out

    @staticmethod
    def get(df, l_limit, h_limit, step):
        start = float(l_limit)
        now = float(start + step)
        to_return = nd.array
        try:
            last_value = df.truncate(after=start).tail(1).values[0]
        except:
            last_value = 0
        to_return.append(last_value)
        while now < h_limit:
            try:
                last_value = df.truncate(before=start, after=now).tail(1).values[0]
            except:
                pass
            to_return.append(last_value)
            start = start + step
            now = now + step
        return to_return

    def getarow(self, index, col):
        value = float(index)
        to_return = np.array([])
        label = '_'.join((str(value), str(col)))
        for k, ts in self.data.iteritems():
            try:
                fromthists = ts[col].truncate(after=value).tail(1).values[0]
            except:
                fromthists = 0.0
            to_return = np.append(to_return, fromthists)
        self.row[label] = pd.Series(to_return)
        

    def splot(self, columns, start, stop, merge=None):
        start = float(start)
        stop = float(stop)
        if len(columns) == 1:
            merge = True
        if self.isSet:
            if merge:
                plt.figure()
                plt.title('merged set')
                for i, col in enumerate(columns):
                    for ds in self.data:
                        name = '_'.join(('ds_merge', ds, col))
                        self.data[ds][col].truncate(before=start, after=stop).plot()
            else:
                fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                for i, col in enumerate(columns):
                    name = '_'.join(('ds_col', col))
                    axes[i].set_title(name)
                    for ds in self.data:
                        self.data[ds][col].truncate(before=start, after=stop).plot(ax=axes[i])
            self.printto(name)
            plt.close()
                        
                
                  
        
        else:
            for col in columns:
                name = '_'.join(('ts', col))
                self.data[col].truncate(before=start, after=stop).plot()
            self.printto(name)

    def mplot(self, columns, start, stop, step=1, merge=None):
        start = float(start)
        stop = float(stop)
        if len(columns) == 1:
            merge = True
        if self.isSet:
            if merge:
                plt.figure()
                name = 'mean_all_columns'
                for col in columns:
                    thisrange = '_'.join((str(start), str(stop), str(step), str(col)))
                    if thisrange not in self.range:
                        self.createrange(thisrange, col, start, stop, step)
                    self.range[thisrange].mean(1).plot(label=col)
                plt.legend(loc='best')
                plt.title(name)
            else:
                fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                for i, col in enumerate(columns):
                    name = '_'.join(('mean', col))
                    thisrange = '_'.join((str(start), str(stop), str(step), str(col)))
                    if thisrange not in self.range:
                        self.createrange(thisrange, col, start, stop, step)
                    self.range[thisrange].mean(1).plot(label=col, ax=axes[i])
                    axes[i].set_title(name)
                    axes[i].legend(loc='best')
            self.printto(name)
            plt.close()

    def sdplot(self, columns, start, stop, step=1, merge=None):
        start = float(start)
        stop = float(stop)
        if len(columns) == 1:
            merge = True
        if self.isSet:
            if merge:
                plt.figure()
                name = 'std_all_columns'
                for col in columns:
                    thisrange = '_'.join((str(start), str(stop), str(step), str(col)))
                    if thisrange not in self.range:
                        self.createrange(thisrange, col, start, stop, step)
                    self.range[thisrange].std(1).plot(label=col)
                plt.legend(loc='best')
                plt.title(name)
            else:
                fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                for i, col in enumerate(columns):
                    name = '_'.join(('std', col))
                    thisrange = '_'.join((str(start), str(stop), str(step), str(col)))
                    if thisrange not in self.range:
                        self.createrange(thisrange, col, start, stop, step)
                    self.range[thisrange].std(1).plot(label=col, ax=axes[i])
                    axes[i].set_title(name)
                    axes[i].legend(loc='best')
            self.printto(name)
            plt.close()


    def msdplot(self, columns, start, stop, step=1, merge=None):
        start = float(start)
        stop = float(stop)
        if len(columns) == 1:
            merge = True
        if self.isSet:
            if merge:
                plt.figure()
                name = 'mean&std_all_columns'
                for col in columns:
                    thisrange = '_'.join((str(start), str(stop), str(step), str(col)))
                    if thisrange not in self.range:
                        self.createrange(thisrange, col, start, stop, step)
                    mean = self.range[thisrange].mean(1)
                    std = self.range[thisrange].std(1)
                    upper = mean + std
                    lower = mean - std
                    mean.plot(label=col)
                    upper.plot(style='k--')
                    lower.plot(style='k--')
                plt.legend(loc='best')
                plt.title(name)
            else:
                fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                for i, col in enumerate(columns):
                    name = '_'.join(('mean&std', col))
                    thisrange = '_'.join((str(start), str(stop), str(step), str(col)))
                    if thisrange not in self.range:
                        self.createrange(thisrange, col, start, stop, step)
                    mean = self.range[thisrange].mean(1)
                    std = self.range[thisrange].std(1)
                    upper = mean + std
                    lower = mean - std
                    mean.plot(label=col, ax=axes[i])
                    upper.plot(style='k--', ax=axes[i])
                    lower.plot(style='k--', ax=axes[i])
                    axes[i].set_title(name)
                    axes[i].legend(loc='best')
            self.printto(name)
            plt.close()

    def itemfreq(self, columns, value, merge=None, bins=None):
        value = float(value)
        if len(columns) == 1:
            merge = True
        if self.isSet:
            if merge:
                plt.figure()
                name = 'item_freq'
                for col in columns:
                    thisrow = '_'.join((str(value), str(col)))
                    if thisrow not in self.row:
                        self.getarow(value, col)
                    if bins:
                        factor = pd.cut(self.row[thisrow], bins)
                        pd.value_counts(factor).sort_index(1).plot(kind='bar', label=col, alpha=0.5)
                    else:
                        valuecounts = self.row[thisrow].value_counts().sort_index(1)
                        valuecounts = valuecounts.reindex(index=np.arange(self.row[thisrow].min(), self.row[thisrow].max()), fill_value=0.0)
                        valuecounts.plot(kind='bar', label=col, alpha=0.5)
                plt.legend(loc='best')
                plt.title(name)
            else:
                fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                for i, col in enumerate(columns):
                    name = '_'.join(('item_freq', col))
                    thisrow = '_'.join((str(value), str(col)))
                    if thisrow not in self.row:
                        self.getarow(value, col)
                    if bins:
                        factor = pd.cut(self.row[thisrow], bins)
                        pd.value_counts(factor).sort_index(1).plot(kind='bar', label=col, alpha=0.5, ax=axes[i])
                    else:
                        valuecounts = self.row[thisrow].value_counts().sort_index(1)
                        valuecounts = valuecounts.reindex(index=np.arange(self.row[thisrow].min(), self.row[thisrow].max()), fill_value=0.0)
                        valuecounts.plot(kind='bar', label=col, ax=axes[i])
                    axes[i].set_title(name)
                    axes[i].legend(loc='best')
            self.printto(name)
            plt.close()

    def relfreq(self, columns, value, merge=None, fit=None, bins=None):
        value = float(value)
        if len(columns) == 1:
            merge = True
        if self.isSet:
            if merge:
                plt.figure()
                name = 'rel_freq'
                for col in columns:
                    thisrow = '_'.join((str(value), str(col)))
                    if thisrow not in self.row:
                        self.getarow(value, col)
                    if bins:
                        factor = pd.cut(self.row[thisrow], bins)
                        valuecounts = pd.value_counts(factor, normalize=True).sort_index(1)
                        valuecounts.plot(kind='bar', label=col, alpha=0.5)
                    else:
                        valuecounts = self.row[thisrow].value_counts(normalize=True).sort_index(1)
                        valuecounts = valuecounts.reindex(index=np.arange(self.row[thisrow].min(), self.row[thisrow].max()), fill_value=0.0)
                        valuecounts.plot(kind='bar', label=col, alpha=0.5)
                    if fit:
                        valuecounts.plot(kind='line', label=col, style='r--')
                plt.legend(loc='best')
                plt.title(name)
            else:
                fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                for i, col in enumerate(columns):
                    name = '_'.join(('rel_freq', col))
                    thisrow = '_'.join((str(value), str(col)))
                    if thisrow not in self.row:
                        self.getarow(value, col)
                    if bins:
                        factor = pd.cut(self.row[thisrow], bins)
                        valuecounts = pd.value_counts(factor, normalize=True).sort_index(1)
                        valuecounts.plot(kind='bar', label=col, alpha=0.5, ax=axes[i])
                    else:
                        valuecounts = self.row[thisrow].value_counts(normalize=True).sort_index(1)
                        valuecounts = valuecounts.reindex(index=np.arange(self.row[thisrow].min(), self.row[thisrow].max()), fill_value=0.0)
                        valuecounts.plot(kind='bar', label=col, ax=axes[i], alpha=0.5)
                    if fit:
                        valuecounts.plot(kind='line', ax=axes[i], style='r--')
                    axes[i].set_title(name)
                    axes[i].legend(loc='best')
            self.printto(name)
            plt.close()

    def printto(self, figname):
        for out in self.outputs:
            if out == 'view':
                plt.show()
            else:
                name = '.'.join((figname, out))
                plt.savefig(name)


def meq_relfreq(df_dict, colname, l_limit, h_limit, step, numbins=10):
    range_df = create_range(df_dict, colname, l_limit, h_limit, step)
    rangeX = np.arange(l_limit, h_limit, step)
    X = np.zeros((len(rangeX),numbins))
    for x in range(len(rangeX)):
        X[x] = rangeX[x]
    Y = []
    Z = []
    for x in rangeX: 
        relfreq, startpoint, binsize, extrap = stats.relfreq(range_df.loc[x].values, numbins=numbins, \
                defaultreallimits=(min(range_df.loc[x]),max(range_df.loc[x])))
        Yline = [startpoint]
        Z.append(list(relfreq))
        for _ in range(1, len(relfreq)):
            next_y = Yline[-1] + binsize
            Yline.append(next_y)
        Y.append(Yline)
    Y = np.array(Y)
    Z = np.array(Z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    cset = ax.contourf(X, Y, Z, alpha=0.5)
    #ax.clabel(cset, fontsize=9, inline=1)
    ax.set_zlim3d(0, 1)
    plt.show()
    return (X, Y, Z)

def meq_itemfreq(df_dict, colname, l_limit, h_limit, step):
    range_df = create_range(df_dict, colname, l_limit, h_limit, step)
    rangeX = np.arange(l_limit, h_limit, step)
    X = np.zeros((len(rangeX),len(rangeX)))
    for x in range(len(rangeX)):
        X[x] = rangeX[x]
    Y = []
    Z = []
    for x in rangeX: 
        itemfreq = stats.itemfreq(range_df.loc[x].values)
        return itemfreq
        Y.append([y[0] for y in itemfreq])
        Z.append([z[1] for z in itemfreq])
    return (X, Y, Z)
    Y = np.array(Y)
    Z = np.array(Z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    cset = ax.contourf(X, Y, Z, cmap=cm.coolwarm)
    ax.clabel(cset, fontsize=9, inline=1)

def rel_pdf(self, df_dict, numbins=10):
    to_return = np.array([])
    for k,v in df_dict.iteritems():
        to_return = np.append(to_return, [k].append(stats.relfreq(v, numbins=numbins)))
    #plt.ion()
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #X, Y, Z = axes3d.get_test_data(0.1)
    #ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
#
#        #for angle in range(0, 360):
#        #    ax.view_init(30, angle)
    #plt.draw()
    return to_return