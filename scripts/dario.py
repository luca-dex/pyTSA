# File per transiente

# import pyTSA
import pytsa as tsa

# Define the path to the dataset folder, and the extension of each file which contains a time-series
FOLDER = './bio_tmp/'

# Define the time instants at which you want to evaluate the probability density function
PDFTIMES = [5, 10, 25, 50, 75, 100]

# Define the time instants at which you want to evaluate the probability density function
MEQTIME_FROM = 0
MEQTIME_TO = 100

# NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'X', 'Y']
NAMES = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

####### Load the dataset
t = tsa.dataset(FOLDER, commentstring='#')

####### Set up the output terminal
t.deloutput('view')
t.addoutput('eps')
t.addoutput('png')
t.addoutput('txt')

####### Plot all the time-series, divided by columns, and plot each column in a different panel
#print('splot: plot all the time-series, divided by columns, and plot each column in a different panel.')
#t.splot()

####### Plot the average and standard deviation (bar plot) of all the time-series, divided by columns, and plot each column in a different panel
print('msdplot: Plot the average and standard deviation (bar plot) of all the time-series, divided by columns, and plot each column in a different panel')
t.msdplot(columns=NAMES, errorbar=True)

####### Plot the probability density function (normalized, with Gaussian fit) of each column at the timed defined
print('pdf: Plot the probability density function of each column at the timed defined')
for times in PDFTIMES:
    t.pdf(times, columns=NAMES, normed=True, fit=True)

####### Plot the master equation (2D normalized) of each column in the time interval defined
print('meq2dPlot the master equation (2D normalized) of each column in the time interval defined')
t.meq2d(columns=NAMES, start=MEQTIME_FROM, stop=MEQTIME_TO, normed=True)

