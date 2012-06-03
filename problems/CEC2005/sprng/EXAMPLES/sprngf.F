C
C                Demonstrates the use of sprng and isprng                
C A random number stream is initialized and a few random double precision 
C numbers and a few integers are printed.  
C
C Uncomment the following line to get the interface with pointer checking
C #define CHECK_POINTERS

       program sprngf
       implicit none

#include "sprng_f.h"

       integer streamnum, nstreams, seed
       SPRNG_POINTER stream
       real*8 rn
       integer irn, i, junk

       streamnum = 0
       nstreams = 1
       seed = 985456376

C   initialize stream 
       stream = init_sprng(streamnum,nstreams,seed,SPRNG_DEFAULT)
       print *, 'Print information about new stream:'
       junk = print_sprng(stream)

       print *, 'Printing 3 random numbers in [0,1):'
       do 100 i = 1, 3
          rn = sprng(stream)
          write(*,"(f8.6)") rn
 100   continue

       print *, 'Printing 3 random integers in [0,2^31):'
       do 200 i = 1, 3
          irn = isprng(stream)
          write(*,"(i16)")  irn
 200   continue

       junk = free_sprng(stream)

       end
