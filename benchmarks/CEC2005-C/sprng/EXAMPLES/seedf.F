C
C                   Demonstrates the use make_sprng_seed                
C 'make_sprng_seed' is used to produce a new seed each time the program is
C  run. Then a few random numbers are printed. 
C
C Uncomment the following line to get the interface with pointer checking
C #define CHECK_POINTERS

       program seedf
       implicit none

#include "sprng_f.h"

       integer streamnum, nstreams, seed, i, junk
       SPRNG_POINTER stream
       real*8 rn

       streamnum = 0
       nstreams = 1
       
C  generate a new seed each time program is run and initialize stream
       seed = make_sprng_seed()
       stream = init_sprng(streamnum,nstreams,seed,SPRNG_DEFAULT)
       print *, 'Printing information about new stream'
       junk = print_sprng(stream)

       print *, 'Printing 3 random numbers in [0,1):'
       do 100 i = 1, 3
          rn = sprng(stream)
          write(*,"(f8.6)") rn
 100   continue

       junk = free_sprng(stream)

       end
