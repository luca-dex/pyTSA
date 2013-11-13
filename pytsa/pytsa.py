# -*- coding: utf-8 -*-

from __future__ import print_function, division
try:
    import builtins as py
except ImportError:
    import __builtin__ as py

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import tables as ts
import sys
import re
from mpl_toolkits.mplot3d import Axes3D
from commentedfile import *
import pydoc as doc

def help(arg = None):
    if arg is None:
        dir()
    else:
        doc.help(arg)

def dataset(path, commentstring=None, colnames=None, delimiter='[\s\t]+', start=-float('inf'), stop=float('inf'), \
    colid=None, ext=None, every=None, numfiles=None, hdf5=None):
    """Return a RedPanda object from a set of time series.

    Keyword arguments:
    path -- where the file is located, in the form '/myworkingdir/timeseries.ext' (required)
    commentstring -- the symbol that indicates a comment line like '#' or '//' (default None)
    colnames -- columns names, in the form ['t', 'col1', 'col2']. If not set the name will be X1, X2 and so on, excluding time axis. (default None)
    delimiter -- a regex. Default is whitespace. (default [\s\t]+)
    start -- first row to import based on time axis
    stop -- last row to import based on time axis
    ext -- extension of the files to be imported, like data or .txt (default None)
    every -- percentage of rows to be loaded, equally distributed over the entire file. 0 is no rows, 1 is the entire file. If Default every row will be loaded (default None)
    numfiles -- in a folder you can load only numfiles files. Files are chosen randomly. (default None)
    """

    # check if pathname is correct
    if not path.endswith('/'):
        path = path + '/'


    #times check
    if start > stop:
        print('maybe start > stop ?\n')
        raise ValueError


    # check if len(colnames) and len(colis) are = 
    if colnames and colid:
        if len(colnames) != len(colid):
            print('colid and colnames must have same length!')
            raise ValueError
    if colnames is None:
        col_pref = 'Y'
    else:
        col_pref = None


    # check if delimiter is ok
    if colid and delimiter != ',':
        print('column selection work only with delimiter = \',\' (yet)')
        raise ValueError

    # if hdf5 create a HDFStore object in 'w' mode
    if hdf5 is None:
        datadict = {}
        hdf5name = None
    else:
        name = hdf5 + '.h5'
        datadict =  pd.HDFStore(name, 'w')  
        r = re.compile(r'[\W]+')
        hdf5name = name 

    # other usefull infos
    timemin = 0.0
    timemax = 0.0
    fileindex = []

    # Only not ending with
    files = [f for f in os.listdir(path) if (os.path.isfile(path + f) )]
    if ext:
        badfiles = [f for f in os.listdir(path) if ((os.path.isfile(path + f) ) and not f.endswith(ext))]
        files = [x for x in files if x not in badfiles]
    if numfiles:
        files = files[:numfiles]


    # progressbar
    numberoffile = len(files)
    size = sum([os.path.getsize(path + f) for f in files]) / (1024**2)
    print ('Files to load: ', numberoffile, ' - Size: {:0.3f}'.format(size), 'Mb')
    progressbarlen = 50
    atraitevery = numberoffile / float(progressbarlen)
    counter = 0.0
    stepcounter = atraitevery
    traitcounter = 0
    if numberoffile >= progressbarlen:
        biggerthanone = True
    else:
        biggerthanone = False

    sys.stdout.write("[%s]" % (" " * progressbarlen))
    sys.stdout.flush()
    sys.stdout.write("\b" * (progressbarlen+1)) # return to start of line, after '['


    # skip dir, parse all file matching ext

    for filename in files:
        actualfile = os.path.join(path, filename)
        datadictname = filename
        if hdf5:
            datadictname = 'f' + r.sub('', datadictname)
        fileindex.append(datadictname)

        # create a fake file and pd.read_csv!
        try:
            source = CommentedFile(open(actualfile, 'rb'), every=every, \
                commentstring=commentstring, low_limit=start, high_limit=stop)
            datadict[datadictname] = pd.read_csv(source, sep=delimiter, index_col=0, \
                header=None, names=colnames, usecols=colid, prefix=col_pref)
            source.close()

        # mmm somethings wrong here
        except ValueError:
            raise
            break

        # maybe commentstring is worng
        except StopIteration:
            sys.stdout.write("\b" * (progressbarlen+2))
            print('Warning! In file', actualfile, 'a line starts with NaN')
            break

        # range limit check
        thismin = datadict[datadictname].index.values.min()
        thismax = datadict[datadictname].index.values.max()
        if thismin < timemin:
            timemin = thismin
        if thismax > timemax:
            timemax = thismax

        # progress bar 
        counter += 1
        if biggerthanone:
            if counter > stepcounter:
                sys.stdout.write("=")
                sys.stdout.flush()
                stepcounter += atraitevery
                traitcounter += 1
        else:
            while stepcounter < int(counter):
                sys.stdout.write("=")
                sys.stdout.flush()
                stepcounter += atraitevery
                traitcounter += 1

    # always progress bar
    if counter == stepcounter:
        sys.stdout.write("=")
        sys.stdout.flush()
        traitcounter += 1
    if traitcounter < progressbarlen:
        sys.stdout.write("=")
        sys.stdout.flush()
    sys.stdout.write("\n")

    # return RedPanda Obj (isset = True)

    return RedPanda(datadict, True, timemin, timemax, fileindex, hdf5name)

def timeseries(path, commentstring=None, colnames=None, delimiter='[\s\t]+', start=-float('inf'), stop=float('inf'), \
    colid=None, every=None):
    '''just one file'''
    
    # microvalidation
    if start > stop:
        print('maybe start > stop ?\n')
    if colnames and colid:
        if len(colnames) != len(colid):
            print('colid and colnames must have same length!')
    if not colnames:
        col_pref = 'Y'
    else:
        col_pref = None

    if colid and delimiter != ',':
        print('column selection work only with delimiter = \',\' (yet)')

    source = CommentedFile(open(path, 'rb'), \
        commentstring=commentstring, low_limit=start, high_limit=stop, every=every)
    timedata = pd.read_csv(source, sep=delimiter, index_col=0, \
        header=None, names=colnames, usecols=colid, prefix=col_pref)
    source.close()

    # return RedPanda Obj (isset = False)
    return RedPanda(timedata, False, timedata.index.values.min(), timedata.index.values.max())

def loadHdf5(path):
    store = ts.openFile(path, 'r')
    table = store.root.info.desc
    isSet = table.attrs.isSet
    timemin = table.attrs.timemin
    timemax = table.attrs.timemax
    fileindex = list(table.attrs.fileindex)

    return RedPanda(store, isSet, timemin, timemax, fileindex, path, newRp = None)



class RedPanda:
    def __init__(self, data, isSet, timemin, timemax, fileindex=None, hdf5name = None, newRp = True):
        # dataset or timeseries
        self.__data = data
        # what's the type
        self.__isSet = isSet
        # start with default terminal (supported type are png, pdf, ps, eps and svg)
        self.__outputs = set()
        self.__outputs.add('view')
        # range -> label:data (pandas df with index)
        self.__range = {}
        self.__row = {}
        self.__timemin = timemin
        self.__timemax = timemax
        self.__hdf5 = hdf5name

        print('Default start value: ', self.__timemin)
        print('Default stop value: ', self.__timemax)
        print('pyTSA data object successfully created, use function \'help()\' \
            to see a list of functions that you can call on this object.')

        if fileindex is not None:
            self.__fileindex = fileindex

        if isSet:
            self.__columns = self.__data[self.__fileindex[0]].columns.values.tolist()
        else:
            self.__columns = self.__data.columns.values.tolist()

        if hdf5name and newRp:
            self.__data.close()
            store = ts.openFile(hdf5name, 'a')
            group = store.createGroup('/', 'info')
            table = store.createTable(group, 'desc', RedPandaInfo)
            table.attrs.isSet = self.__isSet
            table.attrs.timemin = self.__isSet = self.__timemin
            table.attrs.timemax = self.__isSet = self.__timemax
            table.attrs.fileindex =  self.__fileindex
            table.flush()
            store.close()
            

    def getTimemin(self):
        return self.__timemin

    def getTimemax(self):
        return self.__timemax

    def getIndex(self):
        return self.__fileindex

    def getOutputs(self):
        return list(self.__outputs)

    def getColumns(self):
        return self.__columns

    def setTimemin(self, timemin):
        self.__timemin = timemin

    def setTimemax(self, timemax):
        self.__timemax = timemax

    
    def createrange(self, label, colname, start, stop, step):
        """Select 1 column and create a range from start to stop"""
        if not self.__isSet:
            print('createrange works only on dataset')
            return
        index = np.arange(start, stop, step)
        mean_df = pd.DataFrame(index=index)
        for k,v in self.__data.iteritems():
            mean_df.insert(0, k, self.get(v[colname], start, stop, step))
        self.__range[label] = mean_df

    def addoutput(self, out):
        """select outputs from png, pdf, ps, eps and svg"""
        if out in ['png', 'pdf', 'ps', 'eps', 'svg', 'view']:
            self.__outputs.add(out)
        else:
            print(out, 'not in outputs')

    def deloutput(self, out):
        """select outputs from png, pdf, ps, eps and svg"""
        if out in ['png', 'pdf', 'ps', 'eps', 'svg', 'view']:
            try:
                self.__outputs.remove(out)
            except:
                print(out, 'not in outputs')
        else:
            print(out, 'not in outputs')

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
                pass
            to_return.append(last_value)
            start = start + step
            now = now + step
        return to_return

    def getarow(self, index, col):
        value = float(index)
        to_return = np.array([])
        label = '_'.join((str(value), str(col)))
        for k, ts in self.__data.iteritems():
            try:
                fromthists = ts[col].truncate(after=value).tail(1).values[0]
            except:
                fromthists = 0.0
            to_return = np.append(to_return, fromthists)
        self.__row[label] = pd.Series(to_return)
        

    def splot(self, start=None, stop=None, columns=None, merge=None, xkcd=None):
        if self.__hdf5:
            self.__data = pd.HDFStore(self.__hdf5, 'r')
        if start is None:
            start = self.__timemin
        if stop is None:
            stop = self.__timemax
        if columns is None:
            columns = self.__columns
        start = float(start)
        stop = float(stop)
        if len(columns) == 1:
            merge = True

        def internalSplot():
            if self.__isSet:
                if merge:
                    plt.figure()
                    plt.title('merged set')
                    for i, col in enumerate(columns):
                        for ds in self.__fileindex:
                            name = '_'.join(('ds_merge', ds, col, str(start), str(stop)))
                            self.__data[ds][col].truncate(before=start, after=stop).plot()
                else:
                    fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                    

                    for i, col in enumerate(columns):
                        name = '_'.join(('ds_col', col, str(start), str(stop)))
                        axes[i].set_title(name)
                        
                        
                        for ds in self.__fileindex:
                            self.__data[ds][col].truncate(before=start, after=stop).plot(ax=axes[i])       
            else:
                if merge:
                    for col in columns:
                        name = '_'.join(('ts', col, str(start), str(stop)))
                        self.__data[col].truncate(before=start, after=stop).plot()
                else: 
                    fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                    for i, col in enumerate(columns):
                        name = '_'.join(('ds_col', col, str(start), str(stop)))
                        axes[i].set_title(name)
                        self.__data[col].truncate(before=start, after=stop).plot(ax=axes[i])
            self.printto(name)
        
        if (xkcd):
            with plt.xkcd():
                internalSplot()
        else:
            internalSplot()

        
        plt.close()
        if self.__hdf5:
            self.__data.close()

    def mplot(self, start=None, stop=None, columns=None, step=1, merge=None, \
        xkcd=None):
        if start is None:
            start = self.__timemin
        if stop is None:
            stop = self.__timemax
        if columns is None:
            columns = self.__columns
        start = float(start)
        stop = float(stop)
        if len(columns) == 1:
            merge = True

        def internalMplot():
            if self.__isSet:
                if merge:
                    plt.figure()
                    name = 'mean_all_columns'
                    for col in columns:
                        thisrange = '_'.join((str(start), str(stop), str(step), str(col)))
                        if thisrange not in self.__range:
                            self.createrange(thisrange, col, start, stop, step)
                        self.__range[thisrange].mean(1).plot(label=col)
                    plt.legend(loc='best')
                    plt.title(name)
                else:
                    fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                    for i, col in enumerate(columns):
                        name = '_'.join(('mean', col))
                        thisrange = '_'.join((str(start), str(stop), str(step), str(col)))
                        if thisrange not in self.__range:
                            self.createrange(thisrange, col, start, stop, step)
                        self.__range[thisrange].mean(1).plot(label=col, ax=axes[i])
                        axes[i].set_title(name)
                        axes[i].legend(loc='best')
                self.printto(name)

        if (xkcd):
            with plt.xkcd():
                internalMplot()
        else:
            internalMplot()

        plt.clf()
        plt.close()

    def sdplot(self, start=None, stop=None, columns=None, step=1, merge=None, \
        xkcd=None):
        if start is None:
            start = self.__timemin
        if stop is None:
            stop = self.__timemax
        if columns is None:
            columns = self.__columns
        start = float(start)
        stop = float(stop)
        if len(columns) == 1:
            merge = True

        def internasSdplot():    
            if self.__isSet:
                if merge:
                    plt.figure()
                    name = 'std all columns'
                    for col in columns:
                        thisrange = '_'.join((str(start), str(stop), str(step), str(col)))
                        if thisrange not in self.__range:
                            self.createrange(thisrange, col, start, stop, step)
                        self.__range[thisrange].std(1).plot(label=col)
                    plt.legend(loc='best')
                    plt.title(name)
                else:
                    fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                    for i, col in enumerate(columns):
                        name = '_'.join(('std', col))
                        thisrange = '_'.join((str(start), str(stop), str(step), str(col)))
                        if thisrange not in self.__range:
                            self.createrange(thisrange, col, start, stop, step)
                        self.__range[thisrange].std(1).plot(label=col, ax=axes[i])
                        axes[i].set_title(name)
                        axes[i].legend(loc='best')
                self.printto(name)

        if (xkcd):
            with plt.xkcd():
                internalSdplot()
        else:
            internalSdplot()

        plt.close()


    def msdplot(self, start=None, stop=None, columns=None, step=1, merge=None, \
        errorbar=None, bardist=5, xkcd=None):
        if start is None:
            start = self.__timemin
        if stop is None:
            stop = self.__timemax
        if columns is None:
            columns = self.__columns
        start = float(start)
        stop = float(stop)
        if len(columns) == 1:
            merge = True

        def internalMsdplot():    
            if self.__isSet:
                if merge:
                    fig = plt.figure()
                    name = 'mean&std_all_columns'
                    for col in columns:
                        thisrange = '_'.join((str(start), str(stop), str(step), str(col)))
                        if thisrange not in self.__range:
                            self.createrange(thisrange, col, start, stop, step)
                        mean = self.__range[thisrange].mean(1)
                        std = self.__range[thisrange].std(1)
                        mean.plot(label=col)
                        if errorbar:
                            xind = [t for i, t in enumerate(mean.index.values) if (i % bardist) == 0]
                            yval = [t for i, t in enumerate(mean.values) if (i % bardist) == 0]
                            yerr = [t for i, t in enumerate(std.values) if (i % bardist) == 0]
                            plt.errorbar(xind, yval, yerr=yerr, fmt=None)
                        else:
                            upper = mean + std
                            lower = mean - std
                            upper.plot(style='k--', legend=False)
                            lower.plot(style='k--', legend=False)
                    patches, labels = fig.get_axes()[0].get_legend_handles_labels()
                    fig.get_axes()[0].legend(patches[::3], labels[::3], loc='best')
                    plt.title(name)
                else:
                    fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                    for i, col in enumerate(columns):
                        name = '_'.join(('mean&std', col))
                        thisrange = '_'.join((str(start), str(stop), str(step), str(col)))
                        if thisrange not in self.__range:
                            self.createrange(thisrange, col, start, stop, step)
                        mean = self.__range[thisrange].mean(1)
                        std = self.__range[thisrange].std(1)
                        mean.plot(label=col, ax=axes[i])
                        if errorbar:
                            xind = [t for j, t in enumerate(mean.index.values) if (j % bardist) == 0]
                            yval = [t for j, t in enumerate(mean.values) if (j % bardist) == 0]
                            yerr = [t for j, t in enumerate(std.values) if (j % bardist) == 0]
                            axes[i].errorbar(xind, yval, yerr=yerr, fmt=None)
                        else:
                            upper = mean + std
                            lower = mean - std
                            upper.plot(style='k--', ax=axes[i], legend=False)
                            lower.plot(style='k--', ax=axes[i], legend=False)
                        axes[i].set_title(name)
                        handles, labels = axes[i].get_legend_handles_labels()
                        axes[i].legend([handles[0]], [labels[0]], loc='best')
                self.printto(name)

        if (xkcd):
            with plt.xkcd():
                internalMsdplot()
        else:
            internalMsdplot()

        plt.clf()
        plt.close()

    def pdf(self, time, columns=None, merge=None, binsize=None, numbins=None, \
        normed=False, fit=False, range=None, xkcd=None):
        if columns is None:
            columns = self.__columns
        value = float(time)
        if len(columns) == 1:
            merge = True

        def internalPdf(time, binsize, numbins, normed, fit, range):    
            if self.__isSet:
                if merge:
                    plt.figure()
                    name = 'pdf'
                    minrange = None
                    maxrange = None
                    for col in columns:
                        thisrow = '_'.join((str(value), str(col)))
                        if thisrow not in self.__row:
                            self.getarow(value, col)
                        if not minrange or self.__row[thisrow].min() < minrange:
                            minrange = self.__row[thisrow].min()
                        if not maxrange or self.__row[thisrow].max() > maxrange:
                            maxrange = self.__row[thisrow].max()
                    print('range: ', minrange, ' - ', maxrange)
                    if binsize:
                        numbins = int((maxrange - minrange) / binsize)
                    if not numbins:
                        numbins = 10

                    for col in columns:  
                        thisrow = '_'.join((str(value), str(col)))       
                        n, bins, patches = plt.hist(self.__row[thisrow].values, range=[minrange, maxrange], bins=numbins, \
                            normed=normed, alpha=0.5, label=col)
                        if fit:
                            if not normed:
                                print ('Fit only if normed')
                                fit = False
                            else:
                                (mu, sigma) = stats.norm.fit(self.__row[thisrow].values)
                                y = mlab.normpdf(bins, mu, sigma)
                                plt.plot(bins, y, 'r--', linewidth=2)

                    plt.legend(loc='best')
                    plt.title(name)
                else:
                    fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                    for i, col in enumerate(columns):
                        name = '_'.join(('item_freq', col))
                        thisrow = '_'.join((str(value), str(col)))
                        if thisrow not in self.__row:
                            self.getarow(value, col)
                        if binsize:
                            numbins = int((self.__row[thisrow].max() - self.__row[thisrow].min()) / binsize)
                        if not numbins:
                            numbins = 10
                        n, bins, patches = axes[i].hist(self.__row[thisrow].values, bins=numbins, range=range,\
                            normed=normed, alpha=0.75, label=col)
                        
                        if fit:
                            if not normed:
                                print ('Fit only if normed')
                                fit = False
                            else:
                                (mu, sigma) = stats.norm.fit(self.__row[thisrow].values)
                                y = mlab.normpdf(bins, mu, sigma)
                                axes[i].plot(bins, y, 'r--', linewidth = 2)

                        axes[i].set_title(name)
                        axes[i].legend(loc='best')
                self.printto(name)

        if (xkcd):
            with plt.xkcd():
                internalPdf(time, binsize, numbins, normed, fit, range)
        else:
            internalPdf(time, binsize, numbins, normed, fit, range)                
        
        plt.clf()
        plt.close()

    def pdf3d(self, column, moments, binsize=None, numbins=None, normed=False, fit=False, height=None):
        moments = [float(x) for x in moments]
        moments.sort()
        if self.__isSet:
            name = 'pdf'
            minrange = None
            maxrange = None
            for moment in moments:
                thisrow = '_'.join((str(moment), str(column)))
                if thisrow not in self.__row:
                    self.getarow(moment, column)
                if not minrange or self.__row[thisrow].min() < minrange:
                    minrange = self.__row[thisrow].min()
                if not maxrange or self.__row[thisrow].max() > maxrange:
                    maxrange = self.__row[thisrow].max()
            print('range: ', minrange, ' - ', maxrange)
            if binsize:
                numbins = int((maxrange - minrange) / binsize)
            if not numbins:
                numbins = 10

            fig = plt.figure()
            ax = Axes3D(fig)

            for i, moment in enumerate(moments):
                thisrow = '_'.join((str(moment), str(column)))
                histogram, low_range, binsize, extrapoints = stats.histogram(self.__row[thisrow].values, \
                    numbins=numbins, defaultlimits=(minrange, maxrange))
                newx = np.array([low_range + (binsize * 0.5)])
                for index in py.range(1, len(histogram)):
                    newx = np.append(newx, newx[index-1] + binsize)
                

                if normed:
                    histogram = histogram / sum(histogram)

                ax.bar(newx, histogram, zs=i, zdir='y', alpha=0.5, width=binsize, label='A')

                if fit:
                    if not normed:
                        print ('Fit only if normed')
                        fit = False
                    else:
                        (mu, sigma) = stats.norm.fit(self.__row[thisrow].values)
                        gauss = mlab.normpdf(newx, mu, sigma)
                        ax.plot(newx, gauss, zs=i, zdir='y', c='r', ls='--', lw=2)

            if height:
                ax.set_zlim3d(0, height)

            ax.set_xlabel(column)
            ax.set_ylabel('moments')
            ax.set_ylim3d(-1, len(moments))
            yticks = [-1] + py.range(len(moments)) + [len(moments)]
            ytick_labels = [''] + moments + ['']
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels)

            self.printto(name)
            plt.clf()
            plt.close()

    def meq2d(self, start=None, stop=None, columns=None, step=1.0, binsize=None,\
     numbins=None, normed=True, vmax=None):
        if start is None:
            start = self.__timemin
        if stop is None:
            stop = self.__timemax
        if columns is None:
            columns = self.__columns
        step = float(step)
        moments = np.arange(start, stop, step)
        if self.__isSet:
            fig, axes = plt.subplots(nrows=len(columns), ncols=1)
            for i, column in enumerate(columns):
                name = 'meq2d'
                minrange = None
                maxrange = None
                newindex = np.array([])
                thesemoments = []
                for moment in moments:
                    thisrow = '_'.join((str(moment), str(column)))
                    if thisrow not in self.__row:
                        self.getarow(moment, column)
                    thesemoments.append(self.__row[thisrow])
                    if not minrange or self.__row[thisrow].min() < minrange:
                        minrange = self.__row[thisrow].min()
                    if not maxrange or self.__row[thisrow].max() > maxrange:
                        maxrange = self.__row[thisrow].max()
                    newindex = np.append(newindex, self.__row[thisrow].index.values)
                
                newindex = np.unique(np.append(newindex, np.arange(newindex.min(), newindex.max())))
                newindex.sort()
                
                if binsize:
                    numbins = int((maxrange - minrange) / binsize)
                if not numbins:
                    numbins = 10

                histogram, low_range, intbinsize, extrapoints = stats.histogram(thesemoments[0], numbins=numbins, \
                    defaultlimits=(minrange, maxrange))
                if normed:
                    histogram = histogram / sum(histogram)
                
                I = np.array(histogram)
                for j in py.range(1, len(thesemoments)):
                    histogram, low_range, intbinsize, extrapoints = stats.histogram(thesemoments[j], numbins=numbins, \
                        defaultlimits=(minrange, maxrange))
                    if normed:
                        histogram = histogram / sum(histogram)
                    I = np.vstack([I, histogram])

                value = np.array([low_range + (intbinsize * 0.5)])
                for index in py.range(1, len(histogram)):
                    value = np.append(value, value[index-1] + intbinsize)

                if len(columns) == 1:
                    im = axes.imshow(I.T, aspect='auto', interpolation='nearest', \
                        extent=[moments[0], moments[-1], value[0], value[-1]],origin='lower', vmax=vmax)
                    fig.colorbar(im, ax=axes)
                else:    
                    im = axes[i].imshow(I.T, aspect='auto', interpolation='nearest', \
                        extent=[moments[0], moments[-1], value[0], value[-1]],origin='lower', vmax=vmax)
                    fig.colorbar(im, ax=axes[i])

            self.printto(name)
            plt.clf()
            plt.close()



    def meq3d(self, column, start=None, stop=None, step=1.0, binsize=None, numbins=None, normed=True, vmax=None):
            if start is None:
                start = self.__timemin
            if stop is None:
                stop = self.__timemax
            step = float(step)
            moments = np.arange(start, stop, step)
            if self.__isSet:
                fig = plt.figure()
                ax = Axes3D(fig)
                
                name = 'meq3d'
                minrange = None
                maxrange = None
                newindex = np.array([])
                thesemoments = []
                for moment in moments:
                    thisrow = '_'.join((str(moment), str(column)))
                    if thisrow not in self.__row:
                        self.getarow(moment, column)
                    thesemoments.append(self.__row[thisrow])
                    if not minrange or self.__row[thisrow].min() < minrange:
                        minrange = self.__row[thisrow].min()
                    if not maxrange or self.__row[thisrow].max() > maxrange:
                        maxrange = self.__row[thisrow].max()
                    newindex = np.append(newindex, self.__row[thisrow].index.values)
                
                newindex = np.unique(np.append(newindex, np.arange(newindex.min(), newindex.max())))
                newindex.sort()
                
                if binsize:
                    numbins = int((maxrange - minrange) / binsize)
                if not numbins:
                    numbins = 10

                histogram, low_range, intbinsize, extrapoints = stats.histogram(thesemoments[0], numbins=numbins, \
                    defaultlimits=(minrange, maxrange))
                if normed:
                    histogram = histogram / sum(histogram)
                
                I = np.array(histogram)
                for j in py.range(1, len(thesemoments)):
                    histogram, low_range, intbinsize, extrapoints = stats.histogram(thesemoments[j], numbins=numbins, \
                        defaultlimits=(minrange, maxrange))
                    if normed:
                        histogram = histogram / sum(histogram)
                    I = np.vstack([I, histogram])

                value = np.array([low_range + (intbinsize * 0.5)])
                for index in py.range(1, len(histogram)):
                    value = np.append(value, value[index-1] + intbinsize)

                X, Y = np.meshgrid(moments, value)
                surf = ax.plot_surface(X, Y, I.T, rstride=1, cstride=1, cmap=plt.cm.jet, \
                    linewidth=0, antialiased=False)
                fig.colorbar(surf, shrink=0.5, aspect=5)

                ax.set_xlabel('time')
                ax.set_ylabel(column)
                

            self.printto(name)
            plt.clf()
            plt.close()


    def printto(self, figname):
        for out in self.__outputs:
            if out == 'view':
                plt.show()
            else:
                name = '.'.join((figname, out))
                plt.savefig(name)

class RedPandaInfo(ts.IsDescription):
    pass
