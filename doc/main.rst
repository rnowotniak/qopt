Oprogramowanie **RN Optimization Framework** służy do optymalizacji za pomocą najnowocześniejszych
algorytmów ewolucyjnych m.in. QIGA, rQIEA. Założeniem jest elegancja, elastyczność, mały rozmiar.

Elementarne wymagania co do frameworka
======================================

#. **Zmiana evaluatora** (knapsack, TSP, CEC2005, ...)
#. Łatwe wymiana: operatorów, metod inicjalizacji, metod oceny, kryteriów stopy (wszystko na zasadzie slotow, tak jak w PyEvolve)
#. Benchmarki: CEC2005, (CEC2011), knapsack, TSP, ...
#. algorytmy QIGA, rQIEA

Dalej ewentualnie:

* Metody analizy teoretycznej (Banach, bloki budujące)
* Wizualizacja wyników

Chcemy móc korzystać z tego w ten sposób:

::

        from qopt.algorithms import QIGA     # generic QIGA algo
        from qopt.algorithms import bQIGAo   # exact Han's algo
        from qopt.algorithms import bQIGAhe  # H_e gate
        from qopt.algorithms import bQIGAcm  # cross and mut..

        from qopt.problems import knapsack
        from qopt.problems import tsp
        from qopt.problems import sat


        def my_init(alg):
                alg.bestval = -1
                alg._initialize()
                alg._observe()
                alg._repair()
                alg._evaluate()
                alg._storebest()               

        def my_ops(alg):
                ...
                alg._update()
                alg._newXScross(5)
                ...


        newalg = QIGA()
        newalg.initialization += my_init
        newalg.operators += my_ops
        newalg.evaluator = knapsack
        newalg.run()
        print newalg.best

        q2 = bQIGAo()
        q2.evaluator = knapsack
        q2.run()

        q3 = bQIGAcm()
        q3.evaluator = tsp
        q3.run()



::
        
        from qopt.algorithms import QIEA      # generic QIEA algo
        from qopt.algorithms import rQIEA     # Gexiang's rQIEA algo
        from qopt.problems import cec2005
        from qopt.problems import cec2011

        qiea = QIEA()
        qiea.initialization += ...
        qiea.run()

        r = rQIEA()
        r.evaluator = cec2005.f1
        r.run()
        r.evaluator = cec2005.f2
        r.run()



::

        import qopt.framework
        import qopt.algorithms.QIGA as QIGA
        import qopt.problems.knapsack # ...

        q1 = QIGA()
        q1.evaluator = knapsack
        q1.run()
        print q1.best

::

        q2 = QIGA()
        q2.evaluator = tsp
        q2.operators.append(someop)
        q2.run()


Struktura framework'a RN Optimization Framework
===============================================

Zawartość głównego katalogu projektu *RN Optimization Framework* zorganizowana jest następująco:

**CUDA/**
        Implementacje w technologii NVidia CUDA

**EXPERIMENTS/**
        Skrypty z eksperymentami numerycznymi

**PL-GRID/**
        Skrypty związane z implementacjami w środowiskach masowo równoległych (gridowych)

**algorithms/**
        Własne implementacja wybranych algorytmów

**analysis/**
        Skrypty służące do analizy wyników eksperymentów

**contrib/**
        Biblioteki i projekty zewnętrzne, zawierające implementacje różnych algorytmów optymalizacji

**doc/**
        Dokumentacja projektu

**junk/**
        Nieuporządkowane jeszcze fragmenty kodów źródłowych i prób

**problems/**
        Testowe zadania optymalizacji, benchmarki

**tests/**
        Programy testujące poprawność kodu projektu, skrypty testujące przyjęte założenia

**framework.py**
        Zasadniczy kod framework'a

**notatki.txt**
        Luźne notatki związane z projektem

.. todo::
        Przenieść całkowicie zawartość pliku ``notatki.txt`` do niniejszej dokumentacji.
    

Przykłady użycia
================

Importowanie modułów oferowanych przez framework
------------------------------------------------
::

        import qopt
        import qopt.benchmarks.CEC2005.cec2005 as cec2005
        import qopt.analysis.plots #  (?)
        import qopt.analysis.banach
        import qopt.analysis.bblocks
        import qopt.algorithms.rQIEA
        import qopt.algorithms.iQIEA
        import qopt.problems.slam
        import qopt.problems.featuresel
        import qopt.problems.fuzzy
        import qopt.problems.fode
        import qopt.problems.wavelet
        import qopt.problems.elctro
        import qopt.plgrid # (?)
        import qopt.cuda.qiga  # (?)
        import qopt.cuda.gaqpr #  (?)
        import qopt.tools.pyevolve # (?)

        import qopt.algorithms.pyevolve.sga  # ?
        import qopt.algorithms.contrib.firefly #  ?
        import qopt.algorithms.contrib.bat     # ? (cockoo, de, gewa, hs, pso)


Testowe zadania optymalizacyjne
-------------------------------
::

        problems/knapsack
        > ....
        < .....


::

        problems/cec2005 13 3  # f_num dim
        > ....
        < .....

RN Optimization Framework API
=============================

.. automodule:: qopt.framework
    :members:
    :undoc-members:

