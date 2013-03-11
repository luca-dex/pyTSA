#!/usr/bin/env python

import csv
import os

class CommentedFile:
    """ this class skips comment lines. comment lines start with any of the symbols in commentstring """
    def __init__(self, f, commentstring):
        self.f = f
        self.commentstring = commentstring

    # return next line, skip lines starting with commentstring
    def next(self):
        line = self.f.next()
        if self.commentstring:
            while line.startswith(self.commentstring) or len(line.strip())==0:
                line = self.f.next()
        return line

    # return only 'size' lines using next()
    def test_lines(self, size):
        line = self.f.next()
        text = ''
        for _ in range(size):
            while line.startswith(self.commentstring) or len(line.strip())==0:
                line = self.f.next() 
            text += line
            line = self.f.next()
        return text

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

    def set_operation(self, opname, datasetname='default', **kwargs):
        self.check_args(opname, kwargs)
        self.dataset_descriptor[datasetname] = (opname, kwargs)

    def open(self):
        pass

    def testprint(self):
        pass

    @classmethod
    def check_args(self, opname, opargs):
        return True

class SingleDataSetParse(DataSet):
    """ This class analyzes a single file """
    def __init__(self, filename, commentstring=None, delimiter=None, numlines=20, skipinitialspace=True):
        super(SingleDataSetParse, self).__init__(commentstring, delimiter, numlines, skipinitialspace)
        self.filename = filename

    def open(self, datasetname='default'): 
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

    def open(self):
        for filename in self.filenames:
            csvfile = SingleDataSetParse(filename=os.path.join(self.foldername, filename), delimiter=self.delimiter, commentstring=self.commentstring, numlines=self.numlines, skipinitialspace=self.skipinitialspace)
            csvdata = csvfile.open()
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

if __name__ == '__main__':
    
    print '############ FILE SINGOLO #############\n'

    from dataset import SingleDataSetParse as sds
    a = sds('1-state.data', commentstring=('#', '//'), delimiter='\t')
    a.set_operation('prova', datasetname='a')
    a.open('a')
    a.testprint('a')

    print '\n\n############ FILES MULTIPLI #############\n'

    from dataset import MultiDataSetParse as mds
    b = mds('data', commentstring='#', delimiter='\t')
    b.open()
    b.testprint()
