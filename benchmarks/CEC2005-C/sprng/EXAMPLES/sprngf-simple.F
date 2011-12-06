C
C                Demonstrates the use of sprng and isprng                
C A random number stream is initialized and a few random double precision 
C numbers and a few integers are printed.  
C


       program sprngf_simple
       implicit none

#define SIMPLE_SPRNG	
#include "sprng_f.h"

       SPRNG_POINTER junkPtr
       real*8 rn
       integer seed, irn, i, junk

       seed = 985456376


C   initialization
       junkPtr = init_sprng(seed,SPRNG_DEFAULT)
       print *, 'Print information about new stream:'
       junk = print_sprng()

       print *, 'Printing 3 random numbers in [0,1):'
       do 100 i = 1, 3
          rn = sprng()
          write(*,"(f8.6)") rn
 100   continue

       print *, 'Printing 3 random integers in [0,2^31):'
       do 200 i = 1, 3
          irn = isprng()
          write(*,"(i16)")  irn
 200   continue


       end
