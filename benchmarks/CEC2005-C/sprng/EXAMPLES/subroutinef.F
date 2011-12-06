C
C                Demonstrates the use of sprng in subroutines                

       program sprngf
       implicit none

#include "sprng_f.h"

       integer streamnum, nstreams, seed,junk
       SPRNG_POINTER stream

       streamnum = 0
       nstreams = 1
       seed = 985456376

C   initialize stream 
       stream = init_sprng(streamnum,nstreams,seed,SPRNG_DEFAULT)
       print *, 'Printing information about new stream'
       junk = print_sprng(stream)

       call sub1(stream)

       junk = free_sprng(stream)

       end

C --- print random numbers in  subroutine ---
      subroutine sub1(stream)
      
#include "sprng_f.h"
      SPRNG_POINTER stream
      real *8 rn

       print *, 'Printing 3 double precision numbers in [0,1): '
       do 100 i = 1, 3
          rn = sprng(stream)
          write(*, "(i6, 2H   , f19.16)") i, rn
 100   continue

       return 
       end

