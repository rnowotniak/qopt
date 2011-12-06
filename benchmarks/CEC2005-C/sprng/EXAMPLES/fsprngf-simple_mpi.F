C
C          Demonstrates use of the single precision generator      
C One stream is maintained per processor. Each processor prints a few 
C single precision random numbers.
C

       program fsprngf_simple_mpi
       implicit none

#define SIMPLE_SPRNG		! simple interface                        */
#define USE_MPI			! use MPI to find number of processes     */
#define FLOAT_GEN 1	! make 'sprng()' return single precision numbers
#include <mpif.h>
#include "sprng_f.h"

       SPRNG_POINTER stream
       real*4 rn
       integer i, seed, myid, ierror, junk

       call MPI_INIT(ierror)
       call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierror)

       seed = 985456376
C 
       stream = init_sprng(seed,SPRNG_DEFAULT)
       write(*, "('Process',i2,': Print information about stream:')"
     &      ) myid
       junk = print_sprng()

       do 100 i = 1, 3
          rn = sprng()
          write(*, "('Process',i2,', random number',i2,': ',f8.6)")
     &          myid,i,rn
 100   continue

       call MPI_FINALIZE(ierror)

       end
