# qopt
Quantum-Inspired Evolutionary Algorithms for Optimization problems

This repository contains some unpublished before source codes developed by Robert Nowotniak in the years
2010-2015. They were used for research on advanced randomised search
algorithms (mainly quantum-inspired evolutionary and genetic algorithms and
other population methods) for numerical and combinatorial optimisation. 

The programs and algorithms were developed in different programming languages:
C, C++, Python with Cython interfaces, CUDA C kernels, helpers Bash shell scripts and some algorithms even in Matlab.

The source code repository main contents:

* **Algorithms/purepython/** -  algorithms implementation in pure Python (the slowest, initial POC implementations)
* **Algorithms/*.pyx**  -   iQIEA, MyRQIEA2, QIEA1, QIEA2, rQIEA algorithms implementation in Cython
* **C/**  -  some algorithms and test problems implementation in C++
* **CUDA/**  -  CUDA C computational kernels implementing a few algorithms in GPGPUs (superb fast, up to several hundred speedup in multi GPU environment)
* **problems/** -  different numerical optimization functions, knapsack problem, SLAM, SAT (boolean satisfiability problem) encoding different combinatorial problems, also functions from CEC2005, CEC2011, CEC2013 benchmarks
* **EXPERIMENTS/** -  high level repeatable procedures for some experiments (analysis of search domain coverage by schemata, building blocks propagation analysis, histograms charts, speed tests, algorithms convergence analysis etc.)
* **analysis/**   -  auxiliary scripts for results analysis and visualization
* **make**   -  Bash shell script to build all this project
* **test.py**  -  Python script demonstrating how to run some implemented algorithms on a few test problems + results validation
* **contrib/**   -  third-party tools referenced in experiments and copied here
  for convenience.  Caution:
  Copyright for each project in this catalog is independent, and these projects
  are made accessible on their independent licences, chosen by their authors.
  Most of these projects were made available by their authors using the GNU GPL license.

## Examples

Import algorithms from `qopt` and some test problems:

```python
import qopt.algorithms
import qopt.problems
```

Loading test functions / benchmark problems:

```python
# knapsack
knapsack = qopt.problems.knapsack250
print knapsack.evaluate('1' * 250)

# func 1d
f1d = qopt.problems.func1d.f1
print f1d.evaluate2(60.488)

# sat
import qopt.problems._sat
s1 = qopt.problems._sat.SatProblem('problems/sat/flat30-100.cnf')
print s1.evaluate('100100001100100100001100010100010001010010100010100010010010010010001001001100001001001001')

# cec2005
f1 = qopt.problems.CEC2005(1)
print f1.evaluate((0,0))

f1 = qopt.problems.CEC2013(8)
print f1.evaluate(qopt.problems.CEC2013.optimum[:2])
```

Solving combinatorial optimization using a little customized QIGA algorithm (overwriting initialize() and generation() methods):

```python
class QIGA(qopt.algorithms.QIGA):
    def initialize(self):
        super(QIGA, self).initialize()
        print 'my initialization'
        print self.Q

    def generation(self):
        super(QIGA, self).generation()
        if self.t == 5:
            print 'generation %d, bestval: %g' % (self.t, self.bestval)

q = QIGA(chromlen = 250)
q.tmax = 500
q.problem = qopt.problems.knapsack250
import time
t1 = time.time()
for run in xrange(1):
    q.run()
print '100 runs in: %g seconds' % (time.time() - t1)
q.run()

print q.bestval
```

Solving numerical optimization problems using implemented RQIEA algorithm:

```python
# cec2005
r = qopt.algorithms.RQIEA
r.problem = qopt.problems.CEC2005(2)
r.dim = 30
r.bounds = None
r.run()

# cec2011
r.problem = qopt.problems.CEC2011(15)
r.run()
## References
```

## References

The programs collected in this repository were used to conduct research (numerical experiments), whose results were presented
in scientific papers and doctoral dissertation:

1. [Nowotniak, R. and Kucharski, J., 2010. Building blocks propagation in quantum-inspired genetic algorithm. arXiv preprint arXiv:1007.4221.](http://adsabs.harvard.edu/abs/2010arXiv1007.4221N)
2. [Nowotniak, R. and Kucharski, J., 2010. Meta-optimization of quantum-inspired evolutionary algorithm. In Proc. XVII Int. Conf. on Information Technology Systems (Vol. 1, pp. 1-17).](https://www.researchgate.net/profile/Jacek_Kucharski/publication/265099961_Meta-optimization_of_Quantum-Inspired_Evolutionary_Algorithm/links/54da7da60cf261ce15cd4a54.pdf)
3. [Nowotniak, R. and Kucharski, J., 2011. GPU-based massively parallel implementation of metaheuristic algorithms. Automatyka/Akademia Górniczo-Hutnicza im. Stanisława Staszica w Krakowie, 15, pp.595-611.](http://yadda.icm.edu.pl/yadda/element/bwmeta1.element.baztech-article-AGH1-0028-0136)
4. [Nowotniak, R. and Kucharski, J., 2012, GPU-based tuning of quantum-inspired genetic algorithm for a combinatorial optimization problem. Bulletin of the Polish Academy of Sciences: Technical Sciences 60.2: 323-330.](https://www.degruyter.com/view/j/bpasts.2012.60.issue-2/v10175-012-0043-4/v10175-012-0043-4.xml)
5. [Nowotniak, R. and Kucharski, J., 2014. Higher-order quantum-inspired genetic algorithms. In Computer Science and Information Systems (FedCSIS), 2014 Federated Conference on (pp. 465-470). IEEE.](http://ieeexplore.ieee.org/abstract/document/6933052/?reload=true)
6. [**Nowotniak, R., 2015. Analysis of Quantum-Inspired Evolutionary Algorithms (Doctoral dissertation) (in Polish).**](http://robert.nowotniak.com/files/rnowotniak-phd.pdf)
