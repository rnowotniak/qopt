C
C            ____Demonstrates the use of spawn_sprng____          
C A random number stream is initialized and a few random numbers are  
C printed. Then two streams are spawned and a few numbers from one of them
C is printed  
C
C Uncomment the following line to get the interface with pointer checking
C #define CHECK_POINTERS

       program sprng_spawnf
       implicit none

#include "sprng_f.h"

       integer streamnum, nstreams, seed,junk
       integer i, nspawned
       SPRNG_POINTER stream, new(2)
       real*8 rn

       streamnum = 0
       nstreams = 1
       seed = 985456376

C   initialize stream 
       stream = init_sprng(streamnum,nstreams,seed,SPRNG_DEFAULT)
       print *, 'Print information about stream:'
       junk = print_sprng(stream)

       print *, 'Printing 2 random numbers in [0,1):'
       do 100 i = 1, 2
          rn = sprng(stream)
          write(*, "(f8.6)") rn
 100   continue

C   spawn streams 
       print*, 'Spawned two streams'
       nspawned = 2
       nspawned = spawn_sprng(stream, 2, new(1))
       if (nspawned .ne. 2) then
          write(*, "('Error: only ',i3, ' streams  spawned')") nspawned
          call exit(1)
       endif
       print *, 'Information on first spawned stream:'
       junk = print_sprng(new(1))
       print *, 'Information on second spawned stream:'
       junk = print_sprng(new(2))

       print *, 'Printing 2 random numbers from second spawned stream:'
       do 200 i = 1, 2
          rn = sprng(new(2))
          write(*, "(f8.6)") rn
 200   continue

C   free memory
       junk = free_sprng(stream)
       junk = free_sprng(new(1))
       junk = free_sprng(new(2))

       end

