Wprowadzenie
============

Wzajemne wzbogacanie i inspirowanie się dwóch dziedzin: Sztucznej Inteligencji
i Informatyki Kwantowej. Możliwe różne podejścia: 1) Projektowanie elementów
algorytmów kwantowych za pomocą technik ewolucyjnych 2) Rzeczywiste kwantowe
algorytmy ewolucyjne 3) Kwantowo inspirowane algorytmy ewolucyjne (QIEA).

Po raz pierwszy możliwość wprowadzenia dodatkowych elementów losowości,
inspirowanych systemami kwantowymi została zasygnalizowana w pracy [?] w roku
1996. Kolejne publikacje naukowe dotyczące tego obszaru pojawiły się w roku
2000 (Han, Kim) i autorzy tych prac przedstawili początkowy kwantowy
algorytm genetyczny (QIGA).

**kwantowo inspirowane algorytmy genetyczne** (QIGA) oraz **kwantowo inspirowane
algorytmy ewolucyjne** (QIEA)

Problem badawczy: ...

Tezy rozprawy zostały zdefiniowane następująco:

#. aaa
#. bbbb

Celem podjętych badań były: analiza teoretyczna wybranych własności, analiza
wpływu wybranych dyskretnych i rzeczywistych parametrów, doświadczalna
eksploracja przestrzeni (aproksymacja krajobrazu metaprzystosowania), wskazania
jakichś ogólnych wytycznych, stworzenie efektywnych implementacji,
wykorzystujących współczesne środowiska (gridy, GPGPU).

Oryginalne elementy w rozprawie:

#. Nowoczesne techniki metaoptymalizacji
#. Efektywna eksploracja wielowymiarowej przestrzeni parametrów przez implementację w środowisku obliczeń masowo równoległych (CUDA, PL-GRID), kilkusetkrotne przyśpieszenia obliczeń w niektórych przypadkach
#. adaptacja wybranych metod analizy teoretycznej (Banach, bloki budujące)
#. Nowoczesne techniki metaoptymalizacji
#. Metody ekstrakcji wiedzy (uczenie maszynowe)
#. Może także należy spróbować poprawić jakiś najnowszy, najlepszy QEA, stosując techniki, które sprawdzały się dla EAs? Np. wykorzystanie informacji o gradziencie w operatorze genetycznym (np. kąt obrotu, tak jak w DCQGA + dodatkowo np. aproksymacja gradientu czymś tam?) itp.

Informatyka kwantowa
--------------------
Ten rozdział jest maksymalnie skróconym przewodnikiem po podstawach Informatyki
Kwantowej. Ten rozdział powinien być tak napisany, by po jego przeczytaniu
rozumieć podstawy działa komputerów kwantowych i móc czuć się zainspirowanym do
tworzenia nowych, lepszych algorytmów ewolucyjnych; by czuć się zainspirowanym
do wprowadzania nowych elementów losowości do algorytmów ewolucyjnych.

kubit, rejestr kwantowy, bramka kwantowa

.. math::

   W^{3\beta}_{\delta_1 \rho_1 \sigma_2} \approx U^{3\beta}_{\delta_1 \rho_1}

[rys] kubit, rejestr kwantowy, bramka kwantowa

kubit po odczycie przyjmuje jedną z dwóch wartości (jego stan ulega
zniszczeniu); nie ma możliwości odczytu bieżącego stanu (tzn. nie można
odczytać zapisanego w nim rozkładu prawdopodobieństwa)

stany splątane --- definicja, przykład

obwód kwantowy

[rys] przykładowy obwód kwantowy

algorytmy kwantowe: Shor, Grover, kodowanie supergęste, teleportacja kwantowa

Wybrane unikalne własności:

* Nie można odczytać dokładnego stanu kubitu ani rejestru kwantowego
* Wymiar przestrzeni stanów rejestru rośnie w tempie wykładniczym wraz ze wzrostem rozmiaru rejestru (przykład)
* Stany splątane

Algorytmy ewolucyjne
--------------------

Algorytmy ewolucyjne -- jedna z najbardziej znanych gałęzi Sztucznej Inteligencji

Krótka historia AE (SGA,ES,PS,GP), systematyzacja

[rys] klasyfikacja algorytmów ewolucyjnych

**metaoptymalizacja** dobór parametrów algorytmu - zadanie optymalizacji samo w sobie; "*przekleństwo wymiarowości*"

Kwantowo inspirowane algorytmy ewolucyjne
-----------------------------------------

Róznice pomiędzy Informatyką Kwantową a Algorytmami QIEA -- czyli dlaczego
mówimy tu tylko o *inspiracji* i dlaczego te algorytmy nie wymagają
komputera kwantowego?

#. ograniczenie do zbioru liczb rzeczywistych (możliwość reprezentacji na płaszczyźnie)
#. geny w binarnych chromosomach kwantowych są modelowane przez niezależne kubity (nie stanowią one jednego układu kwantowego, złożonego z n podsystemów) -- zatem nie ma tu czegoś takiego jak np. kwantowe splątanie pomiędzy genami
#. jest możliwość odczytania stanu genu kwantowego (bo jest to po prostu wektor w przestrzeni stanów); stan genu kwantowego nie musi być niszczony (i zazwyczaj nie jest)

Metaoptymalizacja
-----------------
CMA-ES

[rys.] Overlayed metaoptimizer, QIEA, problem

[rys.] Krajobraz metaprzystosowania

Implementacje masowo rółnoległe (CUDA, PL-GRID)
-----------------------------------------------
Przeprowadzenie analiz doświadczalnych nad rozważaną nową klasą algorytmów
ewolucyjnych wymagało wykonania czasochłonnych oraz bardzo wymagających pod
względem zasobów obliczeniowych eksperymentów obliczeniowych. Aby umożliwić
przeprowadzenie tych badań w rozsądnym czasie, algorytmy QIEA zostały
zaimplementowane w dostępnych współcześnie środowiskach obliczeń masowo
równoległych.

Ponadto, kwantowo inspirowane algorytmy genetyczne zostały również
zaimplementowane w technologi GPGPU, pozwalającej na wykonywanie obliczeń
dowolnego przeznaczenia na procesorach kart graficznych. Została w tym celu
wykorzystana technologia NVidia CUDA.

GPGPU, PL-Grid

Python, C, C++

