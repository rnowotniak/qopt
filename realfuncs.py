#!/usr/bin/python

import sys, math
from math import cos,sin,pi,sqrt

presets = []

def addPreset(name,expr,xmin=0,xmax=0,ymin=0,ymax=0,dens=0):
    preset = {}
    preset['name'] = name
    preset['expr'] = expr
    preset['xmin'] = xmin
    preset['xmax'] = xmax
    preset['ymin'] = ymin
    preset['ymax'] = ymax
    presets.append(preset)

addPreset("De Jong's function", "-(x*x+y*y)")
addPreset("Axis parallel hyper-ellipsoid function", "-(x*x+2*y*y)")
addPreset("Rotated hyper-ellipsoid function", "-(x*x+x*x+y*y)")
addPreset("Moved axis parallel hyper-ellipsoid function", "-(5*x*x + 5*2*y*y)")
addPreset("Rosenbrock's valley", "-(100*(y-x*x)*(y-x*x)+(1-x)*(1-x))", -2, 2, -2, 2)
addPreset("Rastrigin's function", "-(10*2+ (x*x-10*cos(2*pi*x)) + (y*y-10*cos(2*pi*y)))", -1, 1, -1, 1, 20)
addPreset("Schwefel's function", "-(-x*sin(sqrt(abs(x))) -y*sin(sqrt(abs(y))))", 100, 600, 400, 800, 20)
addPreset("Griewangk's function", "-(x*x/4000+y*y/4000 - cos(x)*cos(y/sqrt(2)) +1)", -5, 5, -5, 5, 20)
addPreset("Sum of different power function", "-(abs(x)*abs(x) + abs(y)*abs(y)*abs(y))", -1, 1, -1, 1)
addPreset("Ackley's Path function", "-(-20.0*exp(-0.2*sqrt(1.0/2.0*(x*x+y*y)))-exp(1.0/2.0*(cos(2.0*pi*x)+cos(2.0*pi*y)))+20.0+exp(1))", -5, 5, -5, 5, 24)
addPreset("Michalewicz's function", "-(-(sin(x)*pow(sin(x*x/pi),(2*10)) +sin(y)*(pow(sin(2.0*y*y/pi),2*10) )))", 1.5, 2.5, 1, 2, 20)
addPreset("Branins's rcos function", "-(1*(y-5.1/(4*pi*pi)*x*x +5.0/pi*x - 6)*(y-5.1/(4*pi*pi)*x*x +5.0/pi*x - 6) + 10*(1-1/8/pi)*cos(x)+10)", -5, 10, 0, 15, 15)
addPreset("Easom's function", "-(-cos(x)*cos(y)*exp(-( (x-pi)*(x-pi) + (y-pi)*(y-pi)  )))", 1, 5, 1, 5, 20)
addPreset("Goldstein-Price's function", "-((1+(x+y+1)*(x+y+1)*(19-14*x+3*x*x-14*y+6*x*y+3*y*y))*(30+(2*x-3*y)*(2*x-3*y)*(18-32*x+12*x*x+48*y-36*x*y+27*y*y)) )", -2, 2, -2, 2, 20)
addPreset("Six-hump camel back function", "-((4-2.1*x*x+pow(x*x*x*x,1.0/3))*x*x+x*y+(-4+4*y*y)*y*y )", -2, 2, -2, 2, 20)
addPreset("Multi criteria function (Rosenbrock+Michalewicz)", " -(100*(y-x*x)*(y-x*x)+(1-x)*(1-x)) -2000*(-(sin(x/30)*pow(sin((x/30)*(x/30)/pi),(2*10)) +sin(y)*pow(sin(2*(y)*(y)/pi),(2*10)) ))", -2, 2, -2, 2)
addPreset("Multi criteria function (Ackley+Michalewicz)", "-(-(sin(x)*pow(sin(x*x/pi),(2*10)) +sin(y)*pow(sin(2*y*y/pi),(2*10)) ))+  -(-20*exp(-0.2*sqrt(1/2*(x*x+y*y)))-exp(1/2*(cos(2*pi*x)+cos(2*pi*y)))+20+exp(1))", -4, 4, -4, 4, 30)


def rosenbrock(x,y,n=2):
    return -((1.0-x)**2 + 100.*(y-x**2)**2)

def simplefunction(x,y):
    return -(abs(x+3)*abs(y-2))


# converters

class Binary:
    def __call__(self,bstr):
        # phenotype = ...
        # val = ...
        return self.xmin + int(bstr, 2) * (self.xmax - self.xmin) / (2**len(bstr) - 1)


class Real:
    def __init__(self):
        self.xmin = 0.
        self.xmax = 0.
        self.ymin = 0.
        self.ymax = 0.
    def __call__(self,bstr):
        xbstr = bstr[:len(bstr)/2]
        ybstr = bstr[len(bstr)/2:]
        x = self.xmin + float(int(xbstr, 2)) * (self.xmax - self.xmin) / (2**len(xbstr) - 1)
        y = self.ymin + float(int(ybstr, 2)) * (self.ymax - self.ymin) / (2**len(ybstr) - 1)
        phenotype = (x,y)
        val = globals()[self.func](*phenotype)
        print phenotype
        return (val, phenotype)

class Eval1D:
    def __init__(self):
        self.xmin, self.xmax = (0., 1.)
    def __call__(self,bstr):
        x = self.xmin + float(int(bstr, 2)) * (self.xmax - self.xmin) / (2**len(bstr) - 1)
        val = eval(self.expr)
        return (val, x)

class Eval2dBin:
    def __init__(self):
        self.preset = 0
        self.xmin, self.xmax = (-1., 1.)
        self.ymin, self.ymax = (-1., 1.)
    def __call__(self,bstr):
        xbstr = bstr[:len(bstr)/2]
        ybstr = bstr[len(bstr)/2:]
        x = self.xmin + float(int(xbstr, 2)) * (self.xmax - self.xmin) / (2**len(xbstr) - 1)
        y = self.ymin + float(int(ybstr, 2)) * (self.ymax - self.ymin) / (2**len(ybstr) - 1)
        phenotype = (x,y)
        val = eval(presets[self.preset]['expr'])
        return (val, phenotype)


class Eval2dReal:
    def __init__(self):
        self.preset = 0
        self.xmin, self.xmax = (-1., 1.)
        self.ymin, self.ymax = (-1., 1.)
    def __call__(self,point):
        x,y = point[:2]
        if len(point) > 2:
            z = point[2]
        else:
            z = 0
        phenotype = point
        preset = presets[self.preset]
        if x < preset['xmin'] or x > preset['xmax'] or y < preset['ymin'] or y > preset['ymax']:
            val = float('-inf')
        else:
            val = eval(preset['expr'])
        return (val, phenotype)

class InterpolatedEvaluator:
    def __init__(self):
        import numpy as np
        self.bspline = None
        #self.knotsx = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        #self.knotsy = np.array([0, 20, 40, 10, 25, 10, 80, 45, 60, 10, 0])
        self.knotsx = np.array([0, 10, 20, 30, 40, 56, 60, 65, 80, 90, 100, 120, 150, 180, 200])
        self.knotsy = np.array([0, 20, 40, 10, 25, 33, 80, 45, 60, 20, 0, 20, 40, 20, 0])
        self.xmin = 0.
        self.xmax = 200.
    def __call__(self,genotype):
        from scipy import interpolate
        if not self.bspline:
            # interpolation knots
            self.bspline = interpolate.splrep(self.knotsx,self.knotsy)
        if type(genotype) == str:
            xbstr = genotype
            x = self.xmin + float(int(xbstr, 2)) * (self.xmax - self.xmin) / (2**len(xbstr) - 1)
        else:
            x = genotype
        return (interpolate.splev(x, self.bspline), x)


if __name__ == '__main__':
    evaluator = InterpolatedEvaluator()
    evaluator('0')
    import numpy as np
    from scipy import interpolate
    import matplotlib.pyplot as plt
    x = np.linspace(evaluator.xmin, evaluator.xmax, 100)
    y = interpolate.splev(x, evaluator.bspline)
    plt.plot(evaluator.knotsx, evaluator.knotsy, 'or', x, y)
    #plt.ylim(0, 150)
    plt.xlim(evaluator.xmin, evaluator.xmax)
    plt.title('Landscape of objective function')
    plt.ylabel('fitness')
    plt.xlabel('x')
    plt.grid(True)
    plt.savefig('/tmp/lala.pdf')
    sys.exit()

    e = Eval2dBin()
    e.preset = 4
    print e('11')
    sys.exit(0)
    # testowanie
    problem = globals()[sys.argv[1]]
    print problem(1,1)

