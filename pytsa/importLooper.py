import os
import pandas as pd
from multiprocessing import Process
from commentedfile import *

class ImportLooper(Process):
    def __init__(self, fileindex, path, queue, re, every, start, stop):
        self.fileindex = fileindex
        self.path = path
        self.queue = queue
        self.re = re
        self.every = every
        self.start = start
        self.stop = stop
        super(ImportLooper, self).__init__()
    def run(self):
        for filename in self.fileindex:
            actualfile = os.path.join(self.path, filename)
            datadictname = filename
            #if self.re:
                #pass
                #datadictname = 'f' + self.re.sub('', datadictname)

            # create a fake file and pd.read_csv!
            try:
                source = CommentedFile(open(actualfile, 'rb'), every=self.every, \
                    commentstring=commentstring, low_limit=self.start, high_limit=self.stop)
                toReturn = pd.read_csv(source, sep=delimiter, index_col=0, \
                    header=None, names=colnames, usecols=colid, prefix=col_pref)
                source.close()

            # mmm somethings wrong here
            except ValueError:
                raise


            # maybe commentstring is worng
            except StopIteration:
                sys.stdout.write("\b" * (progressbarlen+2))
                print('Warning! In file', actualfile, 'a line starts with NaN')


            # range limit check
            #thismin = datadict[datadictname].index.values.min()
            #thismax = datadict[datadictname].index.values.max()
            #if thismin < timemin:
            #    timemin = thismin
            #if thismax > timemax:
            #    timemax = thismax

            que.put((datadictname, toReturn))