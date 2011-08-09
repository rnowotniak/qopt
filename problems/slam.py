#!/usr/bin/python

import getopt
from visual import *
from visual.controls import *
from random import uniform, randint, random
import math, sys, os
import numpy as num
import time
import subprocess

class SLAM:

    def load(self, mapfile, scanfile):
        self.map = [[float(v) for v in l.split(',')] for l in open(mapfile, 'r').readlines()]

        f = open(scanfile, 'r')
        self.x,self.y,self.angle = [float(x) for x in f.readline().split(',')]
        f.readline()
        self.scan = [[float(v) for v in l.split(',')] for l in f.readlines()]

    def visual(self):
        scene = display(title="Robot SLAM", width=930, height=677, x=0,y=0);
        scene.forward = (-0.0490447, -1.08959, 1.02975)
        scene.ambient = 0.4;

        # Grid
        minx = min(self.map, key=lambda xy:xy[0])[0]
        maxx = max(self.map, key=lambda xy:xy[0])[0]
        miny = min(self.map, key=lambda xy:xy[1])[1]
        maxy = max(self.map, key=lambda xy:xy[1])[1]
        stepx = (maxx - minx) / 10
        stepy = (maxy - miny) / 10
        for i in range(11):
            curve(pos=[(stepy*i+miny,0,minx),(i*stepy+miny,0,maxx)], color=color.blue)       
            curve(pos=[(miny,0,stepx*i+minx),(maxy,0,i*stepx+minx)], color=color.blue)
        # axes
        arrow(axis=(1,0,0), color=color.red) # y
        arrow(axis=(0,0,1), color=color.green) # x

        for p in self.map:
            sphere(pos=(p[1],0,p[0]), radius=0.15)

        scanframe = frame()
        robot = arrow(axis=(0,0,3), shaftwidth=5, color=color.yellow, frame=scanframe)
        for p in self.scan:
            sphere(pos=(p[1],0,p[0]), radius=0.15, color=color.red, frame=scanframe)
        scanframe.pos = (self.y,0,self.x)
        scanframe.rotate(angle=self.angle, axis = (0,1,0), origin=(self.y,0,self.x))

        autorotate = False
        view_rotate = 0
        drag = None
        alt = False
        dt = 0.03
        start_time = time.time()
        if saveimg:
            return
        while true:
            end_time = time.time()
            dt = end_time - start_time
            start_time = end_time    
            rate(25);

            # Handle keyboard events
            if scene.kb.keys:
                s = scene.kb.getkey()
                if s == 'q':
                    sys.exit()
                elif s == 'r':
                    autorotate = not autorotate
                elif s == 'c':
                    icp = ICP(self.map, self.scan, 0.13)
                    time1 = time.time()
                    print 'value: %g' % icp.errorValue(self.x, self.y, self.angle)
                    print 'time: %g' % (time.time() - time1)

            # Handle mouse events
            if scene.mouse.events:
                m1 = scene.mouse.getevent()
                if m1.drag:
                    alt = scene.mouse.alt
                    drag = True
                elif m1.drop:
                    drag = False
            if drag:
                if alt:
                    vec1 = scene.mouse.project(normal = scene.up) - scanframe.pos
                    scanframe.rotate(angle=-self.angle, axis = (0,1,0), origin=scanframe.pos)
                    if vec1.z == 0:
                        vec1.z = 1.e-10
                    self.angle = atan(vec1.x / vec1.z)
                    if vec1.z < 0:
                        self.angle = self.angle + pi
                    scanframe.rotate(angle=self.angle, axis = (0,1,0), origin=scanframe.pos)
                else:
                    m1 = scene.mouse.project(normal = scene.up)
                    self.x = m1.z
                    self.y = m1.x
                    scanframe.pos = (self.y,0,self.x)
                print '%f, %f, %f' % (self.x, self.y, self.angle)
            else:
                scene.center = scanframe.pos

            if autorotate:
                view_rotate = view_rotate + 0.1 * math.pi * dt;
                scene.forward = (-1.0 * sin(view_rotate), -1, -1.0 * cos(view_rotate))

class ICP: # Iterative Closest Points
    def __init__(self, scan1, scan2, eps):
        self.scan1 = scan1
        self.scan2 = scan2
        self.eps = eps

    def errorValue(self, x, y, angle):
        temp = self.transform(self.scan2, x, y, angle)
        sum = 0.0
        matches = 0.0
        for p in temp:
            dist = self.nearest(self.scan1, p)
            if (dist <= self.eps):
                sum += dist
                matches += 1
        if matches == 0:
            return float('inf')
        return (sum * len(temp)) / (matches * matches)

    def transform(self, scan, x, y, angle):
        transformed = []
        for p in scan:
            pnew = [0,0]
            pnew[0] = p[0] * cos(angle) - p[1] * sin(angle) + x
            pnew[1] = p[0] * sin(angle) + p[1] * cos(angle) + y
            transformed.append(pnew)
        return transformed

    def nearest(self, scan, p):
        mindist = float('inf')
        for p2 in scan:
            o = (p2[0] - p[0])**2 + (p2[1] - p[1])**2
            mindist = min(o,mindist)
        return sqrt(mindist)

class Evaluator:
    def __init__(self):
        pass
        # self.proc = subprocess.Popen(self.prog)
    def __call__(self, genotype):
        x,y,alpha = genotype
        # self.proc.stdin.write('%g %g %g\n' % (x,y,alpha))
        # self.proc.stdin.flush()
        # val = float(self.proc.stdout.readline())
        #time1 = time.time()
        #self.icp.errorValue(self.x, self.y, self.angle)
        #tdiff = (time.time() - time1)
        return float('-inf')

class BinaryEvaluator:
    def __init__(self):
        self.icp = ICP(map, scan, 0.13)
    def __call__(self):
        # ...
        time1 = time.time()
        self.icp.errorValue(self.x, self.y, self.angle)
        tdiff = (time.time() - time1)

class ExecEvaluator:
    def __init__(self):
        self.cmd = 'slam-evaluator/evaluator'
        self.maxiter = 0
        self.proc = None
        self.xmin, self.xmax = 1.,-1.
        self.ymin, self.ymax = 1.,-1.
        self.zmin, self.zmax = 0.,0.
    def __call__(self,genotype):
        if not self.proc:
            self.proc=subprocess.Popen(
                     self.cmd + ' ' + self.args,
                    shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE, close_fds=True)
        if type(genotype) == str:
            # decode genotype as binary string
            bstr = genotype
            xbstr = bstr[:len(bstr)/3]
            ybstr = bstr[len(bstr)/3:len(bstr)/3*2]
            zbstr = bstr[len(bstr)/3*2:]
            x = self.xmin + float(int(xbstr, 2)) * (self.xmax - self.xmin) / (2**len(xbstr) - 1)
            y = self.ymin + float(int(ybstr, 2)) * (self.ymax - self.ymin) / (2**len(ybstr) - 1)
            alpha = self.zmin + float(int(zbstr, 2)) * (self.zmax - self.zmin) / (2**len(zbstr) - 1)
            phenotype = (x,y,alpha)
        else:
            # parameters are encoded in genotype directly
            phenotype = x,y,alpha = genotype
        self.proc.stdin.write('%f %f %f\n' % (x,y,alpha))
        self.proc.stdin.flush()
        val = float(self.proc.stdout.readline())
        return -val, phenotype

if __name__ == '__main__':
    opts,args=getopt.getopt(sys.argv[1:], '', ['saveimg', 'x=', 'y=', 'alpha='])
    saveimg = False
    for o,a in opts:
        if o=='--x':
            x = float(a)
        elif o=='--y':
            y = float(a)
        elif o=='--alpha':
            alpha = float(a)
        elif o=='--saveimg':
            saveimg = True
    slam = SLAM()
    #slam.load('data/robo_kis_map.txt', 'data/kis_single_scan0.txt')
    slam.load('data/roboMap.txt', 'data/singleScan3.txt')
    if saveimg:
        slam.x, slam.y, slam.angle = x,y,alpha
    slam.visual()
    if saveimg:
        time.sleep(1)
        os.system("scrot /tmp/z.png -e 'mogrify -crop 930x677+1680+20 $f'")
        os.abort()

