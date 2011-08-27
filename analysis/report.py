#!/usr/bin/python
#
# responsibility of this script:
#   run the experiment N times
#

# Example:
# ./report.py [-o /tmp/bla.pdf] ... pso.PSO --dimensions 3 --xmax 50 realfuncs.Rosenbrock ...
# Result:
# /tmp/bla.pdf


from Cheetah.Template import Template
import tempfile
import os,sys
import re,time
import framework,exp
import textwrap
import time
import numpy
import copy
from subprocess import Popen,PIPE
import pickle
import getopt

#import psyco
#psyco.log()
#psyco.profile()


(options,args) = getopt.getopt(sys.argv[1:],
        "o:", ('outdir=', ))
repdir = None
for (o,a) in options:
    if o in ('-o', '--outdir'):
        repdir = a

if repdir:
    try:
        os.mkdir(repdir)
    except OSError, e:
        if e.errno != 17:
            raise
else:
    # create temporary directory
    repdir = tempfile.mkdtemp(prefix='report-')

print 'Directory for the experiment: ' + repdir

time1 = time.time()

# run the experiment
rep = 10 # TODO: brac z argv

best = None
# run the algorithm rep times
for i in xrange(rep):
    cmdline = sys.argv[:]
    cmdline[0] = './exp.py'
    print cmdline
    e = exp.Experiment(cmdline)
    # involve the logging subsystem and run the algorithm
    logging = framework.Logging(repdir+'/log-%d.txt'%i, e.algoobj)
    e.algoobj.run() # run the algorithm
    logging.close()
    if not best or e.algoobj.best.fitness > best.fitness: # minmax issue
        best = copy.deepcopy(e.algoobj.best)

e.best = best
pickle.dump(e, open('%s/exp.pkl'%repdir, 'w'), -1)
#e = pickle.load(open('/tmp/aa.pkl', 'r'))
#import pprint
#pprint.pprint(e.__dict__)
#sys.exit(0)

# calculate the average stats
os.system('python avg.py %s/log-*.txt > %s/avgstats.txt' % (repdir,repdir))

# generate a report based on the template

class Data:
    pass
data=Data()


# various data
data.date=time.strftime('%c')
data.machine = Popen('uname -m', shell=True, stdout=PIPE).communicate()[0]
data.cpu = filter(lambda l: l.startswith('model name'), open('/proc/cpuinfo','r').readlines())[0]
data.cpu = re.sub(r'.*:\s*', '', data.cpu).strip()
data.ram = int(float(Popen(['free'], stdout=PIPE).stdout.readlines()[1].split()[1])/1024)
data.opsys = Popen('uname -o', shell=True, stdout=PIPE).communicate()[0]
data.codepath = os.path.abspath('pso.py')
data.experiment = e
data.framework = framework
data.totalTime = time.time() - time1
data.fulllog = None # open('/tmp/log.txt', 'r').read()
data.rep = rep
data.repdir = repdir
data.best = best

# cmd 
cmd = ''
for a in sys.argv:
    if a.count(' ') or a.count('*'):
        a = "'%s'" % a
    cmd += a + ' '
data.cmd = textwrap.TextWrapper(subsequent_indent='   ', width=60).fill(cmd)

# stats
avgstats = []
f=open('%s/avgstats.txt'%repdir,'r')
while True:
    line = f.readline().strip()
    if not line:
        break
    if line.startswith('#'):
        continue
    avgstats.append(line.split())
f.close()
data.avgstats = avgstats

template = Template(file='templates/report.tex.tmpl', searchList=data)

# statsstats
a=Popen('for p in %s/log-*; do grep "^[0-9]" $p| tail -n 1; done'%repdir, shell=True,stdout=PIPE)
a=a.communicate()[0].strip()
a=a.split('\n')
a =[map(float,b.split()) for b in a]
a =numpy.matrix(a)
statsstats = numpy.ones((4,7))
statsstats[0,:] = numpy.max(a,0)
statsstats[1,:] = numpy.average(a,0)
statsstats[2,:] = numpy.min(a,0)
statsstats[3,:] = numpy.std(a,0)
data.statsstats = [['%g'%elem for elem in row] for row in statsstats.tolist()]
#print statsstats

# generate figures
#os.system('./graph.py -o %s' % repdir+'/fig1.pdf')


# go to temporary directory
os.chdir(repdir)

f=open('rep.tex', 'w')
print >>f, template
f.close()

os.system('pdflatex rep.tex >/dev/null')
os.system('pdflatex rep.tex >/dev/null')
os.system('cp rep.pdf /tmp/report.pdf')

