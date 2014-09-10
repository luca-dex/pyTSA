#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =======
# License
# =======
# 
# pyTSA is distributed under a 3-clause ("Simplified" or "New") BSD
# license. Parts of Pandas, NumPy, SciPy, numpydoc, bottleneck, which all have
# BSD-compatible licenses, are included. Their licenses follow the pandas
# license.
# 
# pyTSA license
# =============
# 
# Copyright (c) 2014, Luca De Sano, Giulio Caravagna
# All rights reserved.
# See LICENSE.txt 

# import xml.etree.ElementTree as etree
import lxml.etree


def read():
	pass

def write_sbrml(name,
		  title,
		  data):
	tit = title.split('# ')[1]
	col_names = title.split('# ')[2].split()
	filename = name + '.xml'
	NSMAP = {None: "http://www.sbrml.org/sbrml/level1/version1"}
	data_result = lxml.etree.Element('sbrml', nsmap=NSMAP) 
	data_result.set("version", '1')
	data_result.set("level", '1')
	data_result.set("creationDate", 'ci va la data')

	ontologyTerms = lxml.etree.SubElement(data_result, 'ontologyTerms')
	lxml.etree.SubElement(ontologyTerms, 'ontologyTerm', attrib={'id': 'term1',
														  		 'term': 'qualcosa',
																 'sourceTermId': 'altro',
																 'ontologyURI': 'uri?'})

	lxml.etree.SubElement(data_result, 'model', attrib={'name': tit,
														  		'sourceURI': 'bho'})

	operations = lxml.etree.SubElement(data_result, 'operations')
	thisop = lxml.etree.SubElement(operations, 'operation', attrib={'id': 'op1',
														  		 	'ontologyTerm': 'bho'})
	lxml.etree.SubElement(thisop, 'method', attrib={'ontologyTerm': 'bho'})
	lxml.etree.SubElement(thisop, 'software', attrib={'name': 'PyTSA',
														  		   'version': '1.6.5',
														  		   'URL': 'https://github.com/luca-dex/pyTSA'})

	result = lxml.etree.SubElement(thisop, 'result')
	resultComponent = lxml.etree.SubElement(result, 'resultComponent', attrib={'id': 'component1'})
	dimensionDescription = lxml.etree.SubElement(resultComponent, 'dimensionDescription')
	compositeDescription = lxml.etree.SubElement(dimensionDescription, 'compositeDescription', attrib={'name': 'time',
																				'indexType': 'time'})
	tupleDescription = lxml.etree.SubElement(compositeDescription, 'tupleDescription')
	for col in col_names[1:]:
		lxml.etree.SubElement(tupleDescription, 'atomicDescription', attrib={'name': col,
																			 'ontologyTerm': '???',
																			 'valueType': 'double'})


	dimension = lxml.etree.SubElement(resultComponent, 'dimension')
	for row in zip(*data):
		cols = [c for c in row]
		compositeValue = lxml.etree.SubElement(dimension, 'compositeValue', attrib={'indexValue': str(cols[0])})
		tupleValue = lxml.etree.SubElement(compositeValue, 'tuple')
		for col in cols[1:]:
			atomicValue = lxml.etree.SubElement(tupleValue, 'atomicValue')
			atomicValue.text = str(col)
  
	with open(filename, 'w') as xml:
		xml.write(lxml.etree.tounicode(data_result, pretty_print=True))

