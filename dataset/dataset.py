#!/usr/bin/env python

import csv

class CommentedFile:
    def __init__(self, f, commentstring):
        self.f = f
        self.commentstring = commentstring

    def next(self):
        line = self.f.next()
        if self.commentstring:
            while line.startswith(self.commentstring) or len(line.strip())==0:
                line = self.f.next()
        return line

    def test_lines(self, size):
        line = self.f.next()
        text = ''
        for _ in range(size):
            while line.startswith(self.commentstring) or len(line.strip())==0:
                line = self.f.next() 
            text += line
            line = self.f.next()
        return text

    def seek(self):
        self.f.seek(0)

    def __iter__(self):
        return self

class DataSetParse:
    def __init__(self, filename, commentstring=None, delimiter=None, numlines=2, skipinitialspace=True):
        self.filename = filename
        self.delimiter = delimiter
        self.numlines = numlines
        self.commentstring = commentstring
        self.skipinitialspace = skipinitialspace

    def open(self):    
        csvfile = CommentedFile(open(self.filename, 'rb'), commentstring=self.commentstring)
        if self.delimiter:
            self.csvreader = csv.reader(csvfile, delimiter=self.delimiter, skipinitialspace=self.skipinitialspace)
        else:
            dialect = csv.Sniffer().sniff(csvfile.test_lines(self.numlines))
            csvfile.seek()
            self.csvreader = csv.reader(csvfile, dialect)

    def stamp(self):
        a = 0
        for row in self.csvreader:
            a = a + 1
            print row
            if a == 50:
                break

if __name__ == '__main__':
    from dataset import DataSetParse as dsp
    a = dsp('1-state.data', commentstring='#', delimiter='\t')
    a.open()
    a.stamp()