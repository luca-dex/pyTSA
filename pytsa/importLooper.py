import os
import pandas as pd
from multiprocessing import Process
from commentedfile import *

class ImportLooper(Process):
    def __init__(self, fileindex, path, queue, re, every, tmin, tmax, commentstring, \
        delimiter, colnames, colid, col_pref):
        self.fileindex = fileindex
        self.path = path
        self.queue = queue
        self.re = re
        self.every = every
        self.tmin = tmin
        self.tmax = tmax
        self.commentstring = commentstring
        self.delimiter = delimiter
        self.colnames = colnames
        self.colid = colid
        self.col_pref = col_pref
        super(ImportLooper, self).__init__()
    def run(self):
        for filename in self.fileindex:
            actualfile = os.path.join(self.path, filename)
            datadictname = filename
            if self.re:
                pass
                datadictname = 'f' + self.re.sub('', datadictname)

            # create a fake file and pd.read_csv!
            try:
                source = CommentedFile(open(actualfile, 'rb'), every=self.every, \
                    commentstring=self.commentstring, low_limit=self.tmin, high_limit=self.tmax)
                toReturn = pd.read_csv(source, sep=self.delimiter, index_col=0, \
                    header=None, names=self.colnames, usecols=self.colid, prefix=self.col_pref)
                source.close()

            # mmm somethings wrong here
            except ValueError:
                raise


            # maybe commentstring is worng
            except StopIteration:
                sys.stdout.write("\b" * (progressbarlen+2))
                print('Warning! In file', actualfile, 'a line starts with NaN')

            self.queue.put((datadictname, toReturn))