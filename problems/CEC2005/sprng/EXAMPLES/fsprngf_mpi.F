C
C          Demonstrates use of the single precision generator      
C One stream is maintained per processor. Each processor prints a few 
C single precision random numbers.
C
C Uncomment the following line to get the interface with pointer checking
C #define CHECK_POINTERS

       program fsprngf_mpi
       implicit none

#define FLOAT_GEN 1	! make 'sprng()' return single precision numbers
#include <mpif.h>
#include "sprng_f.h"

       integer streamnum, nstreams, seed
       SPRNG_POINTER stream
       real*4 rn
       integer i, myid, nprocs, ierror, junk

      call MPI_INIT(ierror)

      call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierror)

      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierror)

       
       streamnum = myid
       nstreams = nprocs
       seed = 985456376
C 
       stream = init_sprng(streamnum,nstreams,seed,SPRNG_DEFAULT)
       write(*, "('Process',i2,': Print information about stream:')"
     &      ) myid
       junk = print_sprng(stream)

       do 100 i = 1, 3
          rn = sprng(stream)
          write(*, "('Process',i2,', random number',i2,': ',f8.6)")
     &          myid,i,rn
 100   continue

       junk = free_sprng(stream)
       call MPI_FINALIZE(ierror)


       end
