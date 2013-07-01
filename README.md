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

<<<<<<< HEAD
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
ds.mplot(columns=[..], from=.., to=.. ) 	#media di tracce
sdplot(ds, columns=[..], from=.., to=.. ) 	#standard deviation di tracce
msdplot(ds, columns=[..], from=.., to=.. ) 	#media + standard deviation di tracce
```
=======

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


////////////////// OUTPUT
terminali sono view, raw, eps, ps, pdf, png color/white
- possibilita di settare piu di un terminale (default view)
- nomi dei file coerenti


gli output al momento funzionanti sono: png, pdf, ps, eps and svg + view(std)









ho dovuto cambiare from -> to in start -> stop perchè from è una keyword

// carica un sottoinsieme delle colonne: colonna 3 e 23, nome 'a'
ts = timeseries('foo.tsv', colid=[3,23], colnames=['a', 'dio'])
-> tanti nomi quanto colonne usate, se i nomi sono presenti.
	ok

problema: sottoinsieme delle colonne funzione solo quando il delimiter è ',' (il problema arriva dal parser C interno a pandas, dovrebbe risolversi con la 0.12 di pandas)
 
////////// Dataset (valgono tutti i discorsi di prima)

ts = dataset('./', ext='csv')
file-> estensione vuota = suffisso assente. Def '.csv'
	ok

modificato in ext: file è keyword

in questo momento l'oggetto contiene i seguenti dati (supponendo il nome ts):

ts.data -> dati importati da file
ts.isSet -> indica se è dataset o timeseries
ts.view -> output (di default lo std su view)
ts.range -> i range creati dall'utente o internamente nelle elaborazioni 
(x accedere direttamente ts.range['label'] dove label è il nome dato in fase di creazione=

//////////// Plot singola traccia
Ipotesi: ts e' un timeseries, ds e' dataset
splot(ts, columns=[..], from=.., to=.. ) // traccia base
err-> colonna non presente
   -> out of range

modificato in ts.splot(param...)

////////// Plot singola traccia dataset
splot(ds, columns=[..], from=.., to=.. ) // traccia base per ogni file del dataset
warn -> troppi plots? interagire con l'utente, o disabilitare le finestre di default
	ok

// traccia base per ogni file del dataset, ma in unico file!
splot(ds, columns=[..], from=.., to=.., merge=True)
	ok

////////// Plot media/deviazione standard
mplot(ds, columns=[..], from=.., to=.. ) // media di tracce
	ok
sdplot(ds, columns=[..], from=.., to=.. ) // standard deviation di tracce
	ok
msdplot(ds, columns=[..], from=.., to=.. ) // media + standard deviation di tracce
	ok

-> riflettere sulla rappresentazione del file raw
-> riflettere sulla possibilita di implementare tutto con delle BITMASK
   mplot(..., AVG | STDV )
   con default: AVG | STDV

////////// Plot medie/... + singole tracce
merge(
	mplot(ds, columns=[..], from=.., to=.. ), // media di tracce
	splot(ds, columns=[..], from=.., to=.., merge )
);
-> merge deve essere attivo.. o lo forzi attivo tu
>>>>>>> 8cd9bac37cd86f464cb47415ab904fc0987b6feb
