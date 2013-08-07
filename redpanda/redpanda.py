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
import sys
from mpl_toolkits.mplot3d import Axes3D
from commentedfile import *

def dataset(path, commentstring=None, colnames=None, delimiter='[\s\t]+', start=-float('inf'), stop=float('inf'), \
    colid=None, ext=None, every=None, numfiles=None):
    '''more than one file'''
    if not path.endswith('/'):
        path = path + '/'
    #microvalidation
    if start > stop:
        print('maybe start > stop ?\n')
    if colnames and colid:
        if len(colnames) != len(colid):
            print('colid and colnames must have same length!')
    if colnames is None:
        col_pref = 'Y'
    else:
        col_pref = None


    # if ext and not ext.startswith('.'):
    #    ext = ''.join(('.', ext))

    if colid and delimiter != ',':
        print('column selection work only with delimiter = \',\' (yet)')

    datadict = {}
    files = [f for f in os.listdir(path) if (os.path.isfile(path + f) )]
    if ext:
        badfiles = [f for f in os.listdir(path) if ((os.path.isfile(path + f) ) and not f.endswith(ext))]
        files = [x for x in files if x not in badfiles]
    if numfiles:
        files = files[:numfiles]
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

        # import
        try:
            source = CommentedFile(open(actualfile, 'rb'), every=every, \
                commentstring=commentstring, low_limit=start, high_limit=stop)
            datadict[filename] = pd.read_csv(source, sep=delimiter, index_col=0, \
                header=None, names=colnames, usecols=colid, prefix=col_pref)
            source.close()

        except ValueError:
            raise
            break

        except StopIteration:
            sys.stdout.write("\b" * (progressbarlen+2))
            print('Warning! In file', actualfile, 'a line starts with NaN')
            break

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



    if counter == stepcounter:
        sys.stdout.write("=")
        sys.stdout.flush()
        traitcounter += 1


    if traitcounter < progressbarlen:
        sys.stdout.write("=")
        sys.stdout.flush()

    sys.stdout.write("\n")

    # return RedPanda Obj (isset = True)
    return RedPanda(datadict, True)

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
    return RedPanda(timedata, None)

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
            print('createrange works only on dataset')
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
            print(out, 'not in outputs')

    def deloutput(self, out):
        """select outputs from png, pdf, ps, eps and svg"""
        if out in ['png', 'pdf', 'ps', 'eps', 'svg', 'view']:
            try:
                self.outputs.remove(out)
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
        for k, ts in self.data.iteritems():
            try:
                fromthists = ts[col].truncate(after=value).tail(1).values[0]
            except:
                fromthists = 0.0
            to_return = np.append(to_return, fromthists)
        self.row[label] = pd.Series(to_return)
        

    def splot(self, columns, start, stop, merge=None, xkcd=None):
        start = float(start)
        stop = float(stop)
        if len(columns) == 1:
            merge = True
        if self.isSet:
            if merge:
                plt.figure()
                if xkcd:
                    plt.xkcd()
                plt.title('merged set')
                for i, col in enumerate(columns):
                    for ds in self.data:
                        name = '_'.join(('ds_merge', ds, col))
                        self.data[ds][col].truncate(before=start, after=stop).plot()
            else:
                fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                if xkcd:
                    print ('xkcd style only work with merge=True!')
                
                for i, col in enumerate(columns):
                    name = '_'.join(('ds_col', col))
                    axes[i].set_title(name)
                    
                    
                    for ds in self.data:
                        self.data[ds][col].truncate(before=start, after=stop).plot(ax=axes[i])
                        
            self.printto(name)
            plt.close()
        else:
            if merge:
                for col in columns:
                    name = '_'.join(('ts', col))
                    self.data[col].truncate(before=start, after=stop).plot()
            else: 
                fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                for i, col in enumerate(columns):
                    name = '_'.join(('ds_col', col))
                    axes[i].set_title(name)
                    self.data[col].truncate(before=start, after=stop).plot(ax=axes[i])
            self.printto(name)
            plt.close()

    def mplot(self, columns, start, stop, step=1, merge=None, xkcd=None):
        start = float(start)
        stop = float(stop)
        if len(columns) == 1:
            merge = True
        if self.isSet:
            if merge:
                plt.figure()
                if xkcd:
                    plt.xkcd()
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
                if xkcd:
                    print ('xkcd style only work with merge=True!')
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

    def sdplot(self, columns, start, stop, step=1, merge=None, xkcd=None):
        start = float(start)
        stop = float(stop)
        if len(columns) == 1:
            merge = True
        if self.isSet:
            if merge:
                plt.figure()
                if xkcd:
                    plt.xkcd()
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
                if xkcd:
                    print ('xkcd style only work with merge=True!')
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


    def msdplot(self, columns, start, stop, step=1, merge=None, errorbar=None, bardist=5, xkcd=None):
        start = float(start)
        stop = float(stop)
        if len(columns) == 1:
            merge = True
        if self.isSet:
            if merge:
                plt.figure()
                if xkcd:
                    plt.xkcd()
                name = 'mean&std_all_columns'
                for col in columns:
                    thisrange = '_'.join((str(start), str(stop), str(step), str(col)))
                    if thisrange not in self.range:
                        self.createrange(thisrange, col, start, stop, step)
                    mean = self.range[thisrange].mean(1)
                    std = self.range[thisrange].std(1)
                    mean.plot(label=col)
                    if errorbar:
                        xind = [t for i, t in enumerate(mean.index.values) if (i % bardist) == 0]
                        yval = [t for i, t in enumerate(mean.values) if (i % bardist) == 0]
                        yerr = [t for i, t in enumerate(std.values) if (i % bardist) == 0]
                        plt.errorbar(xind, yval, yerr=yerr, fmt=None)
                    else:
                        upper = mean + std
                        lower = mean - std
                        upper.plot(style='k--')
                        lower.plot(style='k--')
                plt.legend(loc='best')
                plt.title(name)
            else:
                fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                if xkcd:
                    print ('xkcd style only work with merge=True!')
                for i, col in enumerate(columns):
                    name = '_'.join(('mean&std', col))
                    thisrange = '_'.join((str(start), str(stop), str(step), str(col)))
                    if thisrange not in self.range:
                        self.createrange(thisrange, col, start, stop, step)
                    mean = self.range[thisrange].mean(1)
                    std = self.range[thisrange].std(1)
                    mean.plot(label=col, ax=axes[i])
                    if errorbar:
                        xind = [t for j, t in enumerate(mean.index.values) if (j % bardist) == 0]
                        yval = [t for j, t in enumerate(mean.values) if (j % bardist) == 0]
                        yerr = [t for j, t in enumerate(std.values) if (j % bardist) == 0]
                        axes[i].errorbar(xind, yval, yerr=yerr, fmt=None)
                    else:
                        upper = mean + std
                        lower = mean - std
                        upper.plot(style='k--', ax=axes[i])
                        lower.plot(style='k--', ax=axes[i])
                    axes[i].set_title(name)
                    axes[i].legend(loc='best')
            self.printto(name)
            plt.close()

    def pdf(self, columns, value, merge=None, binsize=None, numbins=None, normed=False, fit=False, range=None, xkcd=None):
        value = float(value)
        if len(columns) == 1:
            merge = True
        if self.isSet:
            if merge:
                plt.figure()
                if xkcd:
                    plt.xkcd()
                name = 'pdf'
                minrange = None
                maxrange = None
                for col in columns:
                    thisrow = '_'.join((str(value), str(col)))
                    if thisrow not in self.row:
                        self.getarow(value, col)
                    if not minrange or self.row[thisrow].min() < minrange:
                        minrange = self.row[thisrow].min()
                    if not maxrange or self.row[thisrow].max() > maxrange:
                        maxrange = self.row[thisrow].max()
                print('range: ', minrange, ' - ', maxrange)
                if binsize:
                    numbins = int((maxrange - minrange) / binsize)
                if not numbins:
                    numbins = 10

                for col in columns:  
                    thisrow = '_'.join((str(value), str(col)))       
                    n, bins, patches = plt.hist(self.row[thisrow].values, range=[minrange, maxrange], bins=numbins, \
                        normed=normed, alpha=0.5, label=col)
                    if fit:
                        if not normed:
                            print ('Fit only if normed')
                            fit = False
                        else:
                            (mu, sigma) = stats.norm.fit(self.row[thisrow].values)
                            y = mlab.normpdf(bins, mu, sigma)
                            l = plt.plot(bins, y, 'r--', linewidth=2)

                plt.legend(loc='best')
                plt.title(name)
            else:
                fig, axes = plt.subplots(nrows=len(columns), ncols=1)
                if xkcd:
                    print ('xkcd style only work with merge=True!')
                for i, col in enumerate(columns):
                    name = '_'.join(('item_freq', col))
                    thisrow = '_'.join((str(value), str(col)))
                    if thisrow not in self.row:
                        self.getarow(value, col)
                    if binsize:
                        numbins = int((self.row[thisrow].max() - self.row[thisrow].min()) / binsize)
                    if not numbins:
                        numbins = 10
                    n, bins, patches = axes[i].hist(self.row[thisrow].values, bins=numbins, range=range,\
                        normed=normed, alpha=0.75, label=col)
                    
                    if fit:
                        if not normed:
                            print ('Fit only if normed')
                            fit = False
                        else:
                            (mu, sigma) = stats.norm.fit(self.row[thisrow].values)
                            y = mlab.normpdf(bins, mu, sigma)
                            l = axes[i].plot(bins, y, 'r--', linewidth=2)


                    axes[i].set_title(name)
                    axes[i].legend(loc='best')
            self.printto(name)
            plt.close()

    def pdf3d(self, column, moments, binsize=None, numbins=None, normed=False, fit=False, range=None, \
        height=None):
        moments = [float(x) for x in moments]
        moments.sort()
        if self.isSet:
            name = 'pdf'
            minrange = None
            maxrange = None
            for moment in moments:
                thisrow = '_'.join((str(moment), str(column)))
                if thisrow not in self.row:
                    self.getarow(moment, column)
                if not minrange or self.row[thisrow].min() < minrange:
                    minrange = self.row[thisrow].min()
                if not maxrange or self.row[thisrow].max() > maxrange:
                    maxrange = self.row[thisrow].max()
            print('range: ', minrange, ' - ', maxrange)
            if binsize:
                numbins = int((maxrange - minrange) / binsize)
            if not numbins:
                numbins = 10

            fig = plt.figure()
            ax = Axes3D(fig)

            for i, moment in enumerate(moments):
                thisrow = '_'.join((str(moment), str(column)))
                histogram, low_range, binsize, extrapoints = stats.histogram(self.row[thisrow].values, \
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
                        (mu, sigma) = stats.norm.fit(self.row[thisrow].values)
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
            plt.close()

    def meq2d(self, columns, start, stop, step=1.0, binsize=None, numbins=None, normed=True, fit=False, range=None, \
        vmax=None):
        step = float(step)
        moments = np.arange(start, stop, step)
        if self.isSet:
            fig, axes = plt.subplots(nrows=len(columns), ncols=1)
            for i, column in enumerate(columns):
                name = 'meq'
                minrange = None
                maxrange = None
                newindex = np.array([])
                thesemoments = []
                for moment in moments:
                    thisrow = '_'.join((str(moment), str(column)))
                    if thisrow not in self.row:
                        self.getarow(moment, column)
                    thesemoments.append(self.row[thisrow])
                    if not minrange or self.row[thisrow].min() < minrange:
                        minrange = self.row[thisrow].min()
                    if not maxrange or self.row[thisrow].max() > maxrange:
                        maxrange = self.row[thisrow].max()
                    newindex = np.append(newindex, self.row[thisrow].index.values)
                
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
            plt.close()



    def meq3d(self, column, start, stop, step=1.0, binsize=None, numbins=None, normed=True, fit=False, range=None, \
            vmax=None):
            step = float(step)
            moments = np.arange(start, stop, step)
            if self.isSet:
                fig = plt.figure()
                ax = Axes3D(fig)
                
                name = 'meq'
                minrange = None
                maxrange = None
                newindex = np.array([])
                thesemoments = []
                for moment in moments:
                    thisrow = '_'.join((str(moment), str(column)))
                    if thisrow not in self.row:
                        self.getarow(moment, column)
                    thesemoments.append(self.row[thisrow])
                    if not minrange or self.row[thisrow].min() < minrange:
                        minrange = self.row[thisrow].min()
                    if not maxrange or self.row[thisrow].max() > maxrange:
                        maxrange = self.row[thisrow].max()
                    newindex = np.append(newindex, self.row[thisrow].index.values)
                
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
            plt.close()


    def printto(self, figname):
        for out in self.outputs:
            if out == 'view':
                plt.show()
            else:
                name = '.'.join((figname, out))
                plt.savefig(name)