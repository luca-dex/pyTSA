====
bio3
====

Documentazione?
===============

Struttura per la parte di analisi dati
--------------------------------------

Per il momento lavoro su csv/tsv/simili:

Importazione di un singolo file contenente una serie temporale con piu' colonne:

	df = biodf.create_df('file-path', commentstring='#', colnames['time', 'A', 'B'], \
	low_limit='', high_limit='')

Importazione di file da una cartella, ogni file una serie temporale 
con piu' colonne (i file devono avere tuttu lo stesso numero di colonne):

	dfs = biodf.create_dfs('folder-path', commentstring='#', colnames['time', 'A', 'B'], \
	low_limit='', high_limit='')

Creazione di un dataframe su range temporale a step fisso, partendo da un dfs:

	range = biodf.create_range(dfs, colname, l_limit, h_limit, step)
	
Visualizzazione della MEQ con normalizzazione tra 0 e 1:
	
	data.meq_relfreq(self, df_dict, colname, l_limit, h_limit, step, numbins=10)
	
Visualizzazione della MEQ senza normalizzazione tra 0 e 1 (ancora da terminare)
	
	data.meq_itemfreq(self, df_dict, colname, l_limit, h_limit, step)

Sul progetto
============

Struttura che il progetto dovrebbe avere
----------------------------------------

1) Selezione della fonde dei dati (load)
2) Selezione delle operazioni da eseguire sui dati (operation)
3) Selezione della visualizzazione (view)
4) Esecuzione delle istruzioni memorizzate (go)
5) Eventuale aggiustamento dei grafici (solo sulla parte javascript)

To do
-----

- [x] trovare un nome più decente al tutto...
- [ ] rendere la struttura simile a quanto riportato su (http://guide.python-distribute.org/creation.html)
- [ ] gestire l'importazione di xml (o json?)
- [ ] open() esegue l'importazione dei file passandoli tutti e immagazzina solamente i dati necessari
- [ ] creare una funzione che prima di open decide le viste necessarie al file e l'operazione da fare su questa vista
- [ ] verificare se può essere utile (http://sbml.org/Software/libSBML/docs/python-api/libsbml-python-reading-files.html)
- [ ] ristrutturare l'esecuzione dei passi
- [ ] riaoganizzare la struttura gerarchica delle cartelle

Idee
----

Nell'archiviazione delle informazioni ho 3 fasi:

- selezione della sorgente
- selezione dell'operazione da fare (indicando paramentri come righe, colonne,...). Posso ripetere più volte questo passaggio
- scorrimento di tutta la sorgente salvando i dati relativi alle varie operazioni. Unica esecuzione per tutte le operazioni stabilite?
- applicazione delle operazioni ai dati
- visualizzazione dei dati. Manca da capire dove definirla e se definirla prima dell'effettiva importazione dei dati


