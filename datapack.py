import os, codecs, json


def prepare(dataloc, datatype):
	source = codecs.open(os.path.join(os.path.dirname(__file__), dataloc), 'r', 'utf-8')
	dim = len(split(source[0]))
	for row in source:
		splitted = row.split()





# in caso di test
if __name__ == '__main__':
	pass