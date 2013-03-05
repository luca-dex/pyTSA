bio3
====
Struttura per la parte di analisi dati
--------------------------------------

Per il momento lavoro su csv/tsv/simili:

- sds -> single data source
- mds -> multiple data source

Esempio:

	data = sds('file-path', commentstring=None, delimiter=None, numlines=20, skipinitialspace=True)
	data.open(datasetname='reader', [scelta di cosa caricare])
	data.testprint(datasetname='reader')
	data.view(datasetname='reader', [scelta di come effettuare la visualizzazione])
	data.display()


