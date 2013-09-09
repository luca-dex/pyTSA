import redpanda as rp

t = rp.dataset('./', commentstring='#', every=0.2, ext='data')

t.deloutput('view')
t.addoutput('png')

print('splot')
t.splot()

print('set timemax to 200')
t.setTimemax(200)

print('splot')
t.splot()

print('splot - merge')
t.splot(merge=True)

print('mplot')
t.mplot()

print('mplot merge')
t.mplot(merge=True)

print('msd plot')
t.msdplot()

print('msdplot errorbar')
t.msdplot(errorbar=True)

print('pdf - 10 - normed - fit')
t.pdf(10, normed=True, fit=True)

print('pdf3D - X1 - 10 20 30')
t.pdf3d('X1', [10, 20, 30])

print('meq2d - start = 0 stop = 50')
t.meq2d(start=0, stop=50)

print('meq3d - X2 - start = 0 stop = 50')
t.meq3d('X2', start=0, stop=50)