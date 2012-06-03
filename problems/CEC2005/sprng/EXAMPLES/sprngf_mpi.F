C
C           Demonstrates sprng use with one stream per process          
C A distinct stream is created on each process then prints a few      
C random numbers.  
C
C Uncomment the following line to get the interface with pointer checking
C #define CHECK_POINTERS

       program sprngf_mpi
       implicit none

#include <mpif.h>
#include "sprng_f.h"

       integer streamnum, nstreams, seed
       SPRNG_POINTER stream
       real*8 rn
       integer i, myid, nprocs, ierror, junk

      call MPI_INIT(ierror)

      call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierror)

      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierror)

       
       streamnum = myid
       nstreams = nprocs
       seed = 985456376
C 
       stream = init_sprng(streamnum,nstreams,seed,SPRNG_DEFAULT)
       write(*,"('Process', i2, ', print information about stream:')")
     &       myid
       junk = print_sprng(stream)

       do 100 i = 1, 3
          rn = sprng(stream)
          write(*, "('Process',i2,', random number ',i1,': ',f16.14)")
     &    myid,i,rn
 100   continue

       junk = free_sprng(stream)
       call MPI_FINALIZE(ierror)

       end
