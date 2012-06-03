C
C              Demonstrates use of shared and non-shared streams    
C Each process has two streams.  One stream is common to all the    
C processes. The other stream is different on each processor. 
C
C Uncomment the following line to get the interface with pointer checking
C #define CHECK_POINTERS

       program twostreamsf_mpi
       implicit none

#include <mpif.h>
#include "sprng_f.h"

       integer streamnum,commNum, nstreams, seed
       SPRNG_POINTER stream, commonStream
       real*8 rn
       integer i
       integer myid, nprocs, ierror
       integer junk

      call MPI_INIT(ierror)

      call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierror)

      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierror)

       
       streamnum = myid     !This stream is different on each proces
       commNum = nprocs     !This stream is common to all processes
       nstreams = nprocs +1 !extra stream is common to all processes
       seed = 985456376

C  This stream is different on each process
       stream = init_sprng(streamnum,nstreams,seed,SPRNG_DEFAULT)
       write(6, 44) myid
 44    format("Process", i2, ": Print information about new stream")
       junk = print_sprng(stream)

C  This stream is identical on each process
       commonStream = init_sprng(commNum,nstreams,seed,SPRNG_DEFAULT)
       write (6, 55) myid
 55    format ("Process", i2, 
     &      ": This stream is identical on all processes")
       junk = print_sprng(commonStream)

       do 100 i = 1, 2
          rn = sprng(stream)
          write(6, 66) myid, i, rn
 100   continue

       do 200 i = 1, 2
          rn = sprng(commonStream)
          write(6, 77) myid, i, rn
 200   continue
 66    format("Process", i2, 
     &   ", random number (distinct stream)",i2,": ", f8.6)
 77    format("Process", i2, ", random number (shared stream)",
     &   i2,": ", f8.6)
       junk = free_sprng(stream)
       junk = free_sprng(commonStream)

       call MPI_FINALIZE(ierror)

       end
