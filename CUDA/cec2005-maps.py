
import cec2005
import sys
import numpy as np

import matplotlib
matplotlib.use('pdf')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter, ScalarFormatter
import matplotlib.pyplot as plt
import numpy as np

cec2005.nreal = 2

#fig = plt.figure(figsize=plt.figaspect(1.5))
X, Y = np.linspace(-100, 100, 30), np.linspace(-100, 100, 30) # generic
# X, Y = np.linspace(0, 100, 30), np.linspace(-100, 0, 30) # F4
# X, Y = np.linspace(78, 82, 30), np.linspace(-52, -47, 30) # F6

for fnum in xrange(1,7):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    if fnum == 4:
        X, Y = np.linspace(0, 100, 30), np.linspace(-100, 0, 30) # F4
    elif fnum == 6:
        X, Y = np.linspace(78, 82, 30), np.linspace(-52, -47, 30) # F6
    else:
        X, Y = np.linspace(-100, 100, 30), np.linspace(-100, 100, 30) # generic

    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((X.shape[0], X.shape[1]))

    for y in xrange(Y.shape[1]):
        m = np.vstack((X[:,y], Y[:,y])).transpose()
        f = getattr(cec2005, 'f%d' % fnum)
        # f = cec2005.f1(m)
        Z[:,y] = f(m)

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.5, antialiased=False)
    cset = ax.contour(X, Y, Z, zdir='z', offset = 0)

    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')

    plt.title('$F_%d(x,y)$'%fnum)

    plt.savefig('/tmp/f_%d.pdf'%fnum, bbox_inches='tight')

#plt.show()


