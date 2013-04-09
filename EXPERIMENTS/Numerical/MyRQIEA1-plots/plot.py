import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.ticker import NullFormatter

#param1, param2, param3, param4 = 20, 10, -50, 25

param1, param2, param3, param4 = -45, 15, 50, 35
param1, param2, param3, param4 = 0, 25, 43, 11
#param1, param2, param3, param4 = 25, 4, 0, 19
#param1, param2, param3, param4 = 25, 19, 0, 4


##################

# the random data
x = np.random.randn(1000)
y = np.random.randn(1000)

nullfmt   = NullFormatter()         # no labels

# definitions for the axes 
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left+width+0.02

left, width = 0.1, 0.7
bottom, height = 0.1, 0.7
bottom_h = left_h = left+width+0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.15]
rect_histy = [left_h, bottom, 0.15, height]

# start with a rectangular Figure
plt.figure(1, figsize=(8,8))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHistx.yaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)
axHisty.xaxis.set_major_formatter(nullfmt)

# the scatter plot:
#axScatter.scatter(x, y)

# now determine nice limits by hand:
binwidth = 0.25
xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
lim = ( int(xymax/binwidth) + 1) * binwidth
lim = 100

axScatter.set_xlim( (-lim, lim) )
axScatter.set_ylim( (-lim, lim) )


x = np.linspace(-100,100,100)
y = mlab.normpdf(x, param1, param2)
x2 = np.linspace(-100,100,100)
y2 = mlab.normpdf(x2, param3, param4)

axHistx.plot(x, y)
axHisty.plot(y2, x2)

x,x2 = np.meshgrid(x,x2)
y,y2 = np.meshgrid(y,y2)

z = y * y2

axScatter.contourf(x,x2,z, cmap='hot')
axScatter.set_xlabel('$x$')
axScatter.set_ylabel('$y$')

axHistx.set_xlim( (-lim,lim) )
axHisty.set_ylim( (-lim,lim) )



axScatter.grid(False)

#BLAAx = np.arange(6)
#BLAAy = np.arange(5)
#BLAAz = BLAAx * BLAAy[:,np.newaxis]
#plt.set_title('Funkcja nieseparowalna, $f(x,y) \\ne f_x(x)f_y{y}$')
import matplotlib.image as mpimg
background = mpimg.imread('Rosenbrock2_04.png')
#axScatter.imshow(background, extent=(-100,100, -100,100))


#plt.show()
plt.savefig('/tmp/MyRQIEA1.pdf', bbox_inches='tight')
