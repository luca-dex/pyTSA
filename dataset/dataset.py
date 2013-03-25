#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

class CommentedFile(file):
    """ this class skips comment lines. comment lines start with any of the symbols in commentstring """
    def __init__(self, f, commentstring=None, low_limit=-float('inf'), high_limit=float('inf')):
        self.f = f
        self.commentstring = commentstring
        self.l_limit = low_limit
        self.h_limit = high_limit

    # return next line, skip lines starting with commentstring
    def next(self):
        line = self.f.next()
        comments = self.commentstring + '\n'
        while line[0] in comments or float(line.split()[0]) < self.l_limit:
            line = self.f.next()
        if  float(line.split()[0]) < self.h_limit:
            return line
        else:
            self.close()
            raise StopIteration
    
    # moves the cursor to the initial position
    def seek(self):
        self.f.seek(0)

    def close(self):
        self.f.close()

    def __iter__(self):
        return self

class DataSet(object):
    """ This is the DataSet model """
    def __init__(self, commentstring=None, delimiter=None, numlines=20, skipinitialspace=True):
        self.delimiter = delimiter
        self.numlines = numlines
        self.commentstring = commentstring
        self.skipinitialspace = skipinitialspace
        self.dataset = {}
        self.dataset_descriptor = {}

    def task(self, opname, datasetname='default', **kwargs):
        self.check_args(opname, kwargs)
        self.dataset_descriptor[datasetname] = (opname, kwargs)

    def load(self):
        pass

    def testprint(self):
        pass

    @classmethod
    def check_args(self, opname, opargs):
        return True

class SingleDataSetParse(DataSet):
    """ This class analyzes a single file """
    def __init__(self, filename, commentstring=None, delimiter=None, 
        numlines=20, skipinitialspace=True):
        super(SingleDataSetParse, self).__init__(commentstring, delimiter,
            numlines, skipinitialspace)
        self.filename = filename

    def load(self, datasetname='default'): 
        csvfile = CommentedFile(open(self.filename, 'rb'), commentstring=self.commentstring)
        if self.delimiter:
            csvreader = csv.reader(csvfile, delimiter=self.delimiter, skipinitialspace=self.skipinitialspace)
        else:
            dialect = csv.Sniffer().sniff(csvfile.test_lines(self.numlines))
            csvfile.seek()
            csvreader = csv.reader(csvfile, dialect)
        
        self.dataset[datasetname] = []
        for row in csvreader:
            self.dataset[datasetname].append(row)
        csvfile.close()
        return self.dataset[datasetname]
        

    def testprint(self, datasetname='default'):
        a = 0
        for row in self.dataset[datasetname]:
            a = a + 1
            print row
            if a == 15:
                break
        print '...\n#### FINE FILE ####\n'

class MultiDataSetParse(DataSet):
    """ This class analyzes all files contained in a specified folder """
    def __init__(self, foldername, commentstring=None, delimiter=None, numlines=20, skipinitialspace=True):
        super(MultiDataSetParse, self).__init__(commentstring, delimiter, numlines, skipinitialspace)
        self.foldername = foldername
        self.filenames = os.listdir(foldername)

    def load(self):
        for filename in self.filenames:
            csvfile = SingleDataSetParse(filename=os.path.join(self.foldername, filename), 
                delimiter=self.delimiter, commentstring=self.commentstring, numlines=self.numlines, 
                skipinitialspace=self.skipinitialspace)
            csvdata = csvfile.load()
            self.dataset[filename] = csvdata

    def testprint(self):
        a = 0
        for filename in self.filenames:
            for row in self.dataset[filename]:
                a = a + 1
                print row
                if a == 15:
                    break
            print '...\n#### FINE FILE ####\n'
            a = 0

class biodf(object):
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
    
    @classmethod
    def meq(self, df_dict, colname, limit_l, limit_h, step):
        # non usare!!!
        limit_l = float(l_limit)
        limit_h = float(h_limit)
        step = float(step)
        index = np.arange(limit_l, limit_h, step)
        df = sp.DataFrame(index = index)
        now = float(start + step)
        to_return = np.array([])
        for k,v in df_dict.iteritems():
            to_return = np.append(to_return, v[colname].truncate(after=value).tail(1).values[0])
        itemfreq = stats.itemfreq(to_return)
        tot = 0
        for x in itemfreq:
            tot += x[1]
        itemfreq = [[x[0], x[1]/tot] for x in itemfreq]
        return 'scherzavo'


if __name__ == '__main__':
    pass
