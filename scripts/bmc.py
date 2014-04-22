# load the library
import pytsa as tsa 


# import a dataset and name the columns
mydata = tsa.dataset( '.' , colnames =[ 'time' , 'Preys' , 'Predators' ], ext='.data')

#
mydata.deloutput('view')

#
mydata.addoutput('png')

# plot data , as it is
mydata.splot( columns =[ 'Preys' , 'Predators'] , stop =1000)
mydata.phspace([ 'Preys' , 'Predators'], stop = 500)
mydata.phspace3d([ 'Preys' , 'Predators', 'Preys'], stop = 500)

# plot the species probabilities at t =100
mydata.aplot(stop=100)
mydata.asdplot(stop=100, merge=True, legend=True)
mydata.aphspace([ 'Preys' , 'Predators'], stop = 500)
mydata.aphspace3d([ 'Preys' , 'Predators', 'Preys'], stop = 500)
mydata.pdf(100 , columns =[ 'Preys' , 'Predators'] , normed = True , fit = True )
mydata.pdf3d('Preys', moments=[10, 20, 30])


# estimate the master equatio in [0 ,100] as a 2 D heatmap
mydata.meq2d( start =0 , stop =100)
mydata.meq3d('Preys', start=0, stop=100)
