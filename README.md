====
bio3
====

Documentazione?
===============

Struttura per la parte di analisi dati
--------------------------------------

Per il momento lavoro su csv/tsv/simili:

- sds -> single data source
- mds -> multiple data source

Esempio:

	data = sds('file-path', commentstring=None, delimiter=None, numlines=20, skipinitialspace=True)
	data.open(datasetname='default', [scelta di cosa caricare])
	data.testprint(datasetname='default')
	data.view(datasetname='default', [scelta di come effettuare la visualizzazione])
	data.display()

Sul progetto
============

To do
-----

- [ ] trovare un nome più decente al tutto...
- [ ] rendere la struttura simile a quanto riportato su (http://guide.python-distribute.org/creation.html)
- [ ] gestire l'importazione di xml
- [ ] open() esegue l'importazione dei file passandoli tutti e immagazzina solamente i dati necessari
- [ ] creare una funzione che prima di open decide le viste necessarie al file e l'operazione da fare su questa vista
- [ ] verificare se può essere utile (http://sbml.org/Software/libSBML/docs/python-api/libsbml-python-reading-files.html)


