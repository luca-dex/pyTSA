RedPanda
========

RedPanda is a [Pandas](https://github.com/pydata/pandas) extension that allow you to work on multiple time series.

The source code is currently hosted on GitHub at https://github.com/luca-dex/RedPanda. For the latest version:

	git clone https://github.com/luca-dex/RedPanda.git
	cd RedPanda

And via easy_install or pip:

	sudo pip install .

For more information and example click [here](https://github.com/luca-dex/RedPanda/wiki)



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
