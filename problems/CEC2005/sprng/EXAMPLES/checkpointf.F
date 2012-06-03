C/****************************************************************************/
C/*                ____Demonstrates checkpointing____                        */
C/* In a new run, this program initializes a random number stream and prints */
C/* a few random numbers. Finally it packs the state of the stream into a    */
C/* file. In a continuation of an old run, this program reads in the state of*/
C/* a stream from a file, prints a few random numbers, and stores the final  */
C/* state of the stream into a file.                                         */
C/****************************************************************************/
C

C Uncomment the following line to get the interface with pointer checking
C #define CHECK_POINTERS

       program checkpointf
       implicit none

#include "sprng_f.h"

       integer streamnum, nstreams, seed
       SPRNG_POINTER stream
       real*8 rn
       integer i, size, junk
       character buffer1(MAX_PACKED_LENGTH), buffer2(MAX_PACKED_LENGTH)
       character outfile*7, infile*7, firstChar*1

       streamnum = 0
       nstreams = 1
       seed = 985456376

C/*********************** Initialize streams *******************************/
C===========================================
C outfile  -- state.X ; infile  -- state.X
C===========================================
       print*, 'Enter file name(length 7) to store final state:'
       read(*, 111) outfile    ! 7 characters please
       print*, 'Enter file name(length 7) to read from:'
       print*, '         (enter 9 for a new run)'
       read(*, 112) infile     ! 7 characters please
       write(firstChar, 113) infile

 111   format(A7)
 112   format(A7)
 113   format(A1)

       if (firstChar .eq. '9') then  ! initialize stream the first time  
          stream = init_sprng(streamnum,nstreams,seed,SPRNG_DEFAULT)
       else                          ! read stream state from the old file
          open(30, file =infile , status = 'old', form = 'unformatted')
          read(30) size
          read(30) buffer1
          stream = unpack_sprng(buffer1)
          close(30)
       endif
       
       print *, 'Printing 5 random numbers in [0,1): '
       do 100 i = 1, 5
          rn = sprng(stream)
          write(6,150) i, rn
 100   continue

 150      format(i1, "  ", f8.6)

       size = pack_sprng(stream, buffer2)
       open(31, file = outfile, status = 'unknown', 
     &      form = 'unformatted')
       write(31) size
       write(31) buffer2
       close(31)

       junk = free_sprng(stream)

       end
