# -*- coding: utf-8 -*-

from .redpanda import *

class Task:
	def __init__(self, taskname):
		self.jobs = []
		self.taskname = taskname

	def addtask(self, func, *args, **kwargs):
		self.jobs.append((func, args, kwargs))

	def exe(self):
		# creo un oggetto temporaneo, faccio timeseries o dataset 
		a = eval(self.jobs[0][0])(*self.jobs[0][1], **self.jobs[0][2])
		# e poi tutte le altre funzioni che stanno in jobs
		for job in self.jobs[1:]:
			 RedPanda.__dict__[job[0]](a, *job[1], **job[2]) 

	@staticmethod
	def start(*tasksk.s):
		for task in tasks:
			task.exe()
