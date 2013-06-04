# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from . import dataset

class redpanda(object):

    @classmethod
    def prepare_dfs(path, commentstring=None, colnames=None, low_limit=-float('inf'), high_limit=float('inf')):
        ds = dataset()

    @classmethod
    def create_dfs(self, path, commentstring=None, colnames=None, low_limit=-float('inf'), high_limit=float('inf')):
        dfs = {}
        for x in os.listdir(path):
            source = CommentedFile(open(os.path.join(path, x), 'rb'), \
                commentstring=commentstring, low_limit=low_limit, high_limit=high_limit)
            dfs[x] = pd.read_csv(source, delimiter='[\s\t]+', index_col=0, \
                header=None, names=colnames)
        return dfs

    @classmethod
    def create_df(self, path, commentstring=None, colnames=None, low_limit=-float('inf'), high_limit=float('inf')):
        source = CommentedFile(open(path, 'rb'), \
            commentstring=commentstring, low_limit=low_limit, high_limit=high_limit)
        return pd.read_csv(source, delimiter='[\s\t]+', index_col=0, header=None, names=colnames)

    @classmethod
    def get(self, df, l_limit, h_limit, step):
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
    
    @classmethod
    def create_range(self, df_dict, colname, l_limit, h_limit, step):
        index = np.arange(l_limit, h_limit, step)
        mean_df = pd.DataFrame(index=index)
        for k,v in df_dict.iteritems():
            mean_df.insert(0, k, biodf.get(v[colname], l_limit, h_limit, step))
        return mean_df
    
    @classmethod
    def pdf(self, df_dict, colname, value):
        value = float(value)
        to_return = np.array([])
        for k,v in df_dict.iteritems():
            to_return = np.append(to_return, v[colname].truncate(after=value).tail(1).values[0])
        itemfreq = stats.itemfreq(to_return)
        tot = 0
        for x in itemfreq:
            tot += x[1]
        itemfreq = [[x[0], x[1]/tot] for x in itemfreq]
        return itemfreq

    @classmethod
    def meq_relfreq(self, df_dict, colname, l_limit, h_limit, step, numbins=10):
        range_df = biodf.create_range(df_dict, colname, l_limit, h_limit, step)
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
        return (X, Y, Z)

    @classmethod
    def meq_itemfreq(self, df_dict, colname, l_limit, h_limit, step):
        range_df = biodf.create_range(df_dict, colname, l_limit, h_limit, step)
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

    @classmethod
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