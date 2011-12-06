C
C                   Demonstrates the use make_sprng_seed                
C 'make_sprng_seed' is used to produce a new seed each time the program is
C  run. Then a few random numbers are printed. 
C

       program seedf_simple
       implicit none

#define SIMPLE_SPRNG	
#include "sprng_f.h"

       integer seed, i, junk
       SPRNG_POINTER junkPtr
       real*8 rn
       
C  generate a new seed each time program is run and initialize stream
       seed = make_sprng_seed()
       junkPtr = init_sprng(seed,SPRNG_DEFAULT)
       print *, 'Printing information about new stream'
       junk = print_sprng()

       print *, 'Printing 3 random numbers in [0,1):'
       do 100 i = 1, 3
          rn = sprng()
          write(*,"(f8.6)") rn
 100   continue


       end
