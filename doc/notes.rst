
Pozostałe rzeczy
================
Inne narzędzia do optymalizacji
-------------------------------

Te narzędzia możemy sobie podpiąć, opakować je przez nasz framework.
Załóżmy już, że ograniczymy się tylko do narzędzi bezpośrednio Pythonowych:

* **PyEvolve** (http://pyevolve.sf.net/) (zainstalowane, w calosci w Pythonie)  <-  podstawowe ES
* **PyBrain** (http://pybrain.org/pages/features) (zainstalowane, w calosci w Pythonie)  <-! dziala ale wolno
* **PyGMO** -- wraper pagmo (http://pagmo.sourceforge.net/pygmo/index.html) (zainstalowane, powinno byc szybsze, jest .so)  <-  cos ciezko z tym idzie.  przyklady z dokumentacji nie dzialaja
* OpenOpt (http://openopt.org/Welcome) (zainstalowane, calkowicie w Pythonie) <- sa rozne dziwne algorytmy ale nie ma podstawowe PSO, CMAES itp
* pyOpt (http://www.pyopt.org/contents.html) (zainstalowane, jest .so) <-!  bardzo dziwne algorytmy wylacznie
* inspyred (http://inspyred.github.com/) -- chyba wymaga Pythona 2.7
* deap (http://code.google.com/p/deap/)  --  w calosci w Pythonie, jest CMAES nie ma PSO

Pozostaje: PyBrain (proste, jest wszystko, ale jest wolne)

To wykluczamy, m.in. z tego powodu, że nie ma potrzeby korzystać z narzędzi
napisanych w innych jezykach (i budować dodatkową warstwę pośredniczącą),
jeśli jest bogactwo narzędzi napisanych w Pythonie (powyżej):

* PaGMO (Parallel Global Multiobjective Optimizer) (http://sourceforge.net/projects/pagmo/) C++
* GABI (bardzo stare)
* Paradiseo (C++)
* ECSPY ( http://code.google.com/p/ecspy/ ) -- deprecated
* Matlab Optimization Toolbox
* Swarmops (Pedersen): C#, C, Java, Matlab
* Jenes (http://jenes.ciselab.org/) Java
* Alglib (http://www.alglib.net/#download) -- inne jezyki niż Python, EA tutaj to tylko margines
* Open BEAGLE (http://beagle.gel.ulaval.ca/) C++
* http://eodev.sourceforge.net/
* http://dev.heuristiclab.com/trac/hl/core
* http://cs.gmu.edu/~eclab/projects/ecj/



