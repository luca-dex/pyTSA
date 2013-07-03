from redpanda import *
# ouptuput


# qualce simulazione
dataset('./', ext='tsv', numfiles=10)
ds.addouput('pdf')
splot()

fullds = dataset('./', ext='tsv')

...




task1 = Task('Full report')
task1.dataset('./', ext='tsv', numfiles=10, dname='tracce')
task1.splot(dname='tracce', mpanels=true)
task1.dataset('./', ext='tsv', dname='full')
task1.msdplot(dname='full', mpanels=true)
task1.pdfplot(dname='full', mpanels=true, every=25 )
task1.meqplot(dname='full', 3D=true)
task1.output('pdf')
task1.output('png')
#task1.output('view')
task.fout('fullrep')
task.merge()
task.exec()









