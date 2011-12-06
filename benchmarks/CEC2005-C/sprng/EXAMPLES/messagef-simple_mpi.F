C
C            Demonstrates passing a stream to another process    
C Process 0 initializes a random number stream and prints a few random  
C numbers. It then passes this stream to process 1, which recieves it  
C and prints a few random numbers from this stream. 

       program messagef_simple_mpi
       implicit none

#define SIMPLE_SPRNG	
#include <mpif.h>
#include "sprng_f.h"

       integer seed,i,myid, nprocs, ierror, len, junk
       integer status(MPI_STATUS_SIZE)
       SPRNG_POINTER junkPtr
       real*8 rn
       integer packed(MAX_PACKED_LENGTH)

      call MPI_INIT(ierror)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierror)
      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierror)

       if (nprocs .lt. 2) then
          print *, 'ERROR: At least 2 processes required'
          call MPI_FINALIZE(ierror)
          call exit(1)
       endif

       seed = 985456376
       if (myid .eq. 0) then
          junkPtr = init_sprng(seed,SPRNG_DEFAULT)
          write(6,"('Process',i2,': Print information about stream:')")
     &              myid
          junk = print_sprng()

          write(6,"('Process',i2,': Print 2 random numbers in [0,1):')")
     &                myid
          do 100 i = 1, 2
             rn = sprng()
             write(6, "('Process',i2, ': ', f8.6)") myid, rn
 100      continue

          len = pack_sprng(packed)
          ! inform process 1 how many bytes process 0 will send.
          call MPI_SEND(len, 1, MPI_INTEGER, 1, 0, MPI_COMM_WORLD,
     &         ierror)
          call MPI_SEND(packed, len, MPI_INTEGER, 1, 0, MPI_COMM_WORLD,
     &         ierror)

          print*, 'Process 0 sends stream to process 1'
       
       elseif (myid .eq. 1) then
          junkPtr = init_sprng(seed,SPRNG_DEFAULT)
          call MPI_RECV(len, 1, MPI_INTEGER, 0, MPI_ANY_TAG, 
     &                  MPI_COMM_WORLD,status, ierror)
          call MPI_RECV(packed, len, MPI_INTEGER, 0, MPI_ANY_TAG,
     &                  MPI_COMM_WORLD, status, ierror)
          junkPtr = unpack_sprng(packed)
          print *, 'Process 1 has received the packed stream'
          write(6,"('Process',i2,': Print information about stream:')")
     &             myid
          junk = print_sprng()
          print *, 'Process 1 prints 2 numbers from received stream:'
          do 200 i = 1, 2
             rn = sprng()
             write(6, "('Process', i2, ': ', f8.6)") myid, rn
 200      continue
            
       endif

       call MPI_FINALIZE(ierror)

        end
