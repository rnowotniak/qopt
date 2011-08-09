#!/usr/bin/python
#
#  procedura badawcza na potrzeby artykulu SLAM na slok
#
# responsibility of this script:
#   compare a set of algorithms for some problem
#
#   popsize = 30 dla wszystkich (rownolicznosc na podstawie http://ecet.ecs.ru.acad.bg/cst05/Docs/cp/SIII/IIIA.1.pdf http://www.waset.org/journals/ijcse/v1/v1-1-6.pdf )
#   pso  c1,c2=2   
#   sga  Pm = ~ 1/(popsize*chromlen)   Pc = 0.8    roulette     converter:binary->real
#   qiga  lookup table    converter:binary->real
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
from plot import plot,readDataFromFile
import glob

#import psyco
#psyco.log()
#psyco.profile()


templatefile = 'templates/cmpreport.tex.tmpl'


# create temporary directory
#repdir = tempfile.mkdtemp(prefix='report-')
repdir = '/tmp/cmpdir'
print 'Temporary directory for the experiment: ' + repdir

time1 = time.time()

# run the experiment
rep = 10 # TODO: brac z argv

# tutaj powinna byc zapisana formalnie procedura badawcza.

# best = None
# # algos = [[sga.SGA,Pc,09],[pso.PSO,c1,2.5]]
# # postexps = []
# # for algo in algos:
# # run the algorithm rep times
# for i in xrange(rep):
#     cmdline = sys.argv[:]
#     # -> utworzenie prawidlowego exp cmdline
#     # cmdline ma byc ['', 'sga.SGA', '--Pc', '0.9', 'slam.ExecEvaluator', '...']
#     e = exp.Experiment(cmdline)
#     # involve the logging subsystem and run the algorithm
#     logging = framework.Logging(repdir+'/log-%d.txt'%i, e.algoobj) # %s-log-%d.txt % algo[0]
#     e.algoobj.run() # run the algorithm
#     logging.close()
#     if not best or e.algoobj.best.fitness > best.fitness: # minmax issue
#         best = copy.deepcopy(e.algoobj.best)
# postexps.append(e)
# calculate the average stats
#os.system('python avg.py %s/log-*.txt > %s/avgstats.txt' % (repdir,repdir))

# generate a report based on the template

exps = {}
exps['pso.PSO'] = pickle.load(open('/tmp/report-pso.PSO/exp.pkl', 'r'))
exps['sga.SGA'] = pickle.load(open('/tmp/report-sga.SGA/exp.pkl', 'r'))
exps['qiga.QIGA'] = pickle.load(open('/tmp/report-qiga.QIGA/exp.pkl', 'r'))
exps['gaqpr.GAQPR'] = pickle.load(open('/tmp/report-gaqpr.GAQPR/exp.pkl', 'r'))
exps['icp.ICP'] = pickle.load(open('/tmp/report-icp.ICP/exp.pkl', 'r'))

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
data.exps = exps
data.framework = framework
data.totalTime = time.time() - time1
data.fulllog = None # open('/tmp/log.txt', 'r').read()
data.repdir = repdir
data.rep = rep

# cmd 
cmd = ''
for a in sys.argv:
    if a.count(' ') or a.count('*'):
        a = "'%s'" % a
    cmd += a + ' '
cmd = sys.argv[0] + ' ' + ' '.join(exps.keys()) + " slam.ExecEvaluator --xmin -12 --xmax 80 --ymin -70 --ymax 34 --zmin -3.14 --zmax 3.14 --args '--map-file data/roboMap.txt --scan-file data/singleScan3.txt --const-treshold'"
data.cmd = textwrap.TextWrapper(subsequent_indent='   ', width=60).fill(cmd)

# stats
stats={}
for k in exps.keys():
    foo = []
    f=open('%s/avgstats.txt'%'/tmp/report-%s'%k,'r')
    while True:
        line = f.readline().strip()
        if not line:
            break
        if line.startswith('#'):
            continue
        foo.append(line.split())
    f.close()
    stats[k] = foo
data.stats = stats

# create the report template
template = Template(file=templatefile, searchList=data)

statsstats={}
for k in exps.keys():
    files=glob.glob('/tmp/report-%s/log-*'%k)
    a=numpy.matrix(numpy.zeros((len(files),7)))
    for i in xrange(len(files)):
        p = files[i]
        a[i,:]=readDataFromFile(p)[-1,:]
    foo = numpy.ones((4,7))
    foo[0,:] = numpy.max(a,0)
    foo[1,:] = numpy.average(a,0)
    foo[2,:] = numpy.min(a,0)
    foo[3,:] = numpy.std(a,0)
    foo = [['%g'%elem for elem in row] for row in foo.tolist()]
    statsstats[k] = foo
data.statsstats = statsstats


# go to temporary directory
os.chdir(repdir)

for k in exps.keys():
    plot1=plot('/tmp/report-%s/avgstats.txt'%k, 0, 3, 4, 5)
    plot1.xlabel('generation')
    plot1.ylabel('fitness')
    plot1.title('')
    plot1.xlim([0,130])
    plot1.legend(('best', 'avg', 'worst'), loc='lower right')
    plot1.grid(True)
    plot1.savefig('%s.pdf'%k.replace('.','-'), bbox_inches='tight')
    plot1.cla()

# comparison plot
import pylab
plot1.cla()
pylab.grid(True)
#for k in exps:
stats=readDataFromFile('/tmp/report-qiga.QIGA/avgstats.txt')
pylab.plot(stats[:,0],stats[:,3],'ro-', label='QIGA')
stats=readDataFromFile('/tmp/report-gaqpr.GAQPR/avgstats.txt')
pylab.plot(stats[:,0],stats[:,3],'rs-', label='GAQPR')
stats=readDataFromFile('/tmp/report-pso.PSO/avgstats.txt')
pylab.plot(stats[:,0],stats[:,3],'.-', label='PSO')
stats=readDataFromFile('/tmp/report-sga.SGA/avgstats.txt')
pylab.plot(stats[:,0],stats[:,3],'x-', label='SGA')
stats=readDataFromFile('/tmp/report-icp.ICP/avgstats.txt')
pylab.plot(stats[:,0],stats[:,3],'+-', label='ICP')
pylab.legend(loc='lower right')
pylab.xlabel('generation')
pylab.ylabel('fitness')
#pylab.title('Comparison plot')
pylab.xlim([0,130])
#pylab.ylim([-11000,-5000])
pylab.ylim([-70,-57])
pylab.savefig('cmp-gen.pdf', bbox_inches='tight')

plot1.cla()
pylab.grid(True)
for k in exps:
    stats=readDataFromFile('/tmp/report-%s/avgstats.txt'%k)
    pylab.plot(stats[:,1],stats[:,3],'x-', label=k)
pylab.legend(loc='lower right')
pylab.xlabel('evaluations')
pylab.ylabel('fitness')
pylab.title('Comparison plot')
pylab.axis('auto')
#pylab.xlim([0,130])
#pylab.ylim(ymin=-2800)
pylab.savefig('cmp-evals.pdf', bbox_inches='tight')

plot1.cla()
pylab.grid(True)
for k in exps:
    stats=readDataFromFile('/tmp/report-%s/avgstats.txt'%k)
    pylab.plot(stats[:,2],stats[:,3],'x-', label=k)
pylab.legend(loc='lower right')
pylab.xlabel('time (ms)')
pylab.ylabel('fitness')
pylab.title('Comparison plot')
pylab.axis('auto')
#pylab.ylim(ymin=-2800)
pylab.savefig('cmp-time.pdf', bbox_inches='tight')


f=open('rep.tex', 'w')
print >>f, template
f.close()

#os.system('cp rep.tex /tmp/rep.tex')
os.system('pdflatex rep.tex >/dev/null')
os.system('pdflatex rep.tex >/dev/null')
os.system('cp rep.pdf /tmp/report.pdf')

