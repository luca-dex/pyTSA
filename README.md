RedPanda
========

Documentazione
--------------

Qualche riga giusto per comprendere il funzionamento del sw scritto. Estensione a panda per lavorare su serie
temporali di dati prevalentemente di tipo biologico.

Come come ottenere il software:

	git clone https://github.com/luca-dex/RedPanda.git
	cd RedPanda
	sudo pip install .

**Struttura per la parte di analisi dati**

Importare la libreria:
```python
from redpanda import *
```

**Serie temporali**

Carica un singolo file temporale.
```python
ts = timeseries('foo.tsv')
```

Carica un file singolo. Delimiter esplicito (default: [\t\s]+ )
```python
ts = timeseries('foo.tsv', delimiter=',')
```

Carica un file singolo, con commento.
```python
ts = timeseries('foo.tsv', commentstring='#')
```

Carica un file singolo, con nomi. (tanti nomi quanto colonne presenti)
```python
ts = timeseries('foo.tsv', colnames=['a','b'] )
```

Carica con intervallo [0,100]
```python
ts = timeseries('foo.tsv', colnames=['a','b'], start=0, stop=100 )
```

Carica un sottoinsieme delle colonne: colonna 3 e 23, nome 'a', 'dio' (funziona solo se delimiter=',')
```python
ts = timeseries('foo.tsv', colid=[3,23], colnames=['a', 'dio'])
```

**Dataset**

Dataset (valgono tutti i discorsi di prima) E' possibile specificare l'estensione
```python
ts = dataset('./', ext='csv')
```

**Definizione dell'output**

E' possibile scegliere il tipo di output tra view, png, pdf, ps, eps e svg. Caricato di default: view
```python
ts.addoutput('output_mode')
```

Per rimuovere un canale di output
```python
ts.deloutput('output_mode')
```

**Plot**

Ipotesi: ts e' un timeseries, ds e' dataset

Plot singola traccia timeseries
```python
ts.splot(columns=[..], start=.., stop=.. )
```
Plot singola traccia dataset
```python
ds.splot(columns=[..], start=.., stop=.. )
```

Traccia base per ogni file del dataset, ma in unico file!
```python
ds.splot(columns=[..], start=.., stop=.., merge=True)
```

**Plot media/deviazione standard**
```python
ds.mplot(columns=[..], start=.., stop=.. ) 	#media di tracce
sdplot(ds, columns=[..], start=.., stop=.. ) 	#standard deviation di tracce
msdplot(ds, columns=[..], start=.., stop=.. ) 	#media + standard deviation di tracce
```

**Uso con gestione dei task**

```python
from redpanda import Task
task1 = Task('task description')
task1.addtask('timeseries', 'foo.tsv')
task1.addtask('splot', columns=['a'], start=.., stop=..)
task1.exe()
```

Se sono presente più task
```python
Task.start(task1, task2)
```
