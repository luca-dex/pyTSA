pyTSA
=====

Many dynamical systems are often analyzed by performing simulation ensembles under different
parameter configurations. Automatizing time-series analysis is key to save time and focus on other
modeling tasks. *pyTSA* is an open source Python tool to make data-analysis as intuitive as possible.
Its scripts can be processed in a pipeline with any simulation tool outputting time-series, and intuitive
commands allow to perform complex analyses.

Current *pyTSA* version supports the following plots, in many graphical formats: single traces
(single panel or multi panel), average and standard deviation of a dataset (with barplot or traces,
single panel or multi panel), 2D/3D probability density function of a quantity at some specific time
point (with normalization and gaussian fit), 2D/3D time-varying probability density function of a
quantity in a time-interval (heatmap or surface) and 2D/3D phase-space.

pyTSA is a [Pandas](https://github.com/pydata/pandas) extension that allow you to work on multiple time series.

The source code is currently hosted on GitHub at https://github.com/luca-dex/pyTSA. For the latest version:

	git clone https://github.com/luca-dex/pyTSA.git
	cd pyTSA

And via easy_install or pip:

	sudo pip install .

For more information and examples click [here](https://github.com/luca-dex/pyTSA/wiki).

There is also a [getting started](https://github.com/luca-dex/pyTSA/wiki/Getting-Started).