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

