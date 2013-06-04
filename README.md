RedPanda
========

Documentazione?
---------------

Qualche riga giusto per comprendere il funzionamento del sw scritto. Estensione a panda per lavorare su serie
temporali di dati prevalentemente di tipo biologico.

Come come ottenere il software:

> git clone https://github.com/luca-dex/RedPanda.git
> python setup.py install

**Struttura per la parte di analisi dati**


Per il momento lavoro su csv/tsv/simili:

Importare la libreria:
```python
from redpanda import *
```

Importazione di un singolo file contenente una serie temporale con piu' colonne:

```python
df = create_df('file-path', commentstring='#', colnames['time', 'A', 'B'], low_limit='', high_limit='')
```

Importazione di file da una cartella, ogni file una serie temporale 
con piu' colonne (i file devono avere tutti lo stesso numero di colonne):

```python
dfs = create_dfs('folder-path', commentstring='#', colnames['time', 'A', 'B'], low_limit='', high_limit='')
```

Creazione di un dataframe su range temporale a step fisso, partendo da un dfs:

```python
new_range = create_range(dfs, colname, l_limit, h_limit, step)
```

Visualizzazione della MEQ con normalizzazione tra 0 e 1:

```python
meq_relfreq(dfs, colname, l_limit, h_limit, step, numbins=10)
```

Visualizzazione della MEQ senza normalizzazione tra 0 e 1 (ancora da terminare)

```python
meq_itemfreq(dfs, colname, l_limit, h_limit, step)
```

Sul progetto
------------

**Struttura che il progetto dovrebbe avere**

1. Selezione della fonde dei dati (load)
2. Selezione delle operazioni da eseguire sui dati (operation)
3. Selezione della visualizzazione (view)
4. Esecuzione delle istruzioni memorizzate (go)
5. Eventuale aggiustamento dei grafici (solo sulla parte javascript)

**To do**

* [x] trovare un nome più decente al tutto...
* [x] rendere la struttura simile a quanto riportato su (http://guide.python-distribute.org/creation.html)
* [ ] gestire l'importazione di xml (o json?)
* [ ] open() esegue l'importazione dei file passandoli tutti e immagazzina solamente i dati necessari
* [ ] creare una funzione che prima di open decide le viste necessarie al file e l'operazione da fare su questa vista
* [ ] verificare se può essere utile (http://sbml.org/Software/libSBML/docs/python-api/libsbml-python-reading-files.html)
* [x] ristrutturare l'esecuzione dei passi
* [x] riaoganizzare la struttura gerarchica delle cartelle
