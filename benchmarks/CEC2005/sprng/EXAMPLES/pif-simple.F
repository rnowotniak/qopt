C/***************************************************************************/
C/*       ____Demonstrates SPRNG use for Monte Carlo integration____        */
C/* Compute pi using Monte Carlo integration. Random points are generated   */
C/* in a square of size 2. The value of pi is estimated as four times       */
C/* the proportion of samples that fall within a circle of unit radius.     */
C/***************************************************************************/

       program pif_simple
       implicit none

#define SIMPLE_SPRNG       ! simple interface 
#include "sprng_f.h"

       integer in, n, in_old, n_old, count_in_circle
       real*8 pi, error, stderror, p, EXACT_PI
       character filename*7

       EXACT_PI = 3.141592653589793238462643
       p = EXACT_PI/4.0
       
C-- initialization: to initialize n, in_old, n_old, and filename
       call initialize(n, in_old, n_old, filename)
         
      in = count_in_circle(n)         ! count samples in circle
      in = in + in_old                ! # in circle, in all runs 
      n  = n + n_old                   ! # of samples, in all runs
      pi = (4.0*in)/n
      error = abs(pi - EXACT_PI)
      stderror = 4*sqrt(p*(1.0-p)/n)
      write(*,114) pi, n
      write(*, 115) error, stderror
 114  format('pi is estimated as ', f18.16,  ' from ', i12, ' samples.')
 115  format('Error = ', g18.8, ' standard error = ', g18.8)

      call save_state(filename, in, n)  !check-point final state
      end

C-- count # of samples in cirlce
      integer function count_in_circle(n) 
      implicit none

#include "sprng_f.h"

      integer n, i, in
      real*8 x, y

      in = 0
      do 10 i=1, n
         x =  2*sprng() - 1.0           !  x coordinate
         y =  2*sprng() - 1.0           !  y coordinate
         if (x*x + y*y .lt. 1.0) then   !check if point (x,y) is in circle
            in = in +1
         endif 
 10   continue
      count_in_circle = in
	
      return 
      end

C-- initialization --
      subroutine initialize(n, in_old, n_old, filename)
      implicit none
#include "sprng_f.h"

      integer n, in_old, n_old, seed, size, junk
      SPRNG_POINTER junkPtr
      character filename*7, firstChar*1, buffer(MAX_PACKED_LENGTH)

       print*, 'Enter 9 for a new run, or 2 to continue an old run'
       read(*,113) firstChar
       print*, 'Enter file name(length 7) to store final state:'
       read(*, 111) filename    ! 7 characters please
       print*, 'Enter number of new samples:'
       read(*,*)n

 111   format(A7)
 113   format(A1)

      if (firstChar .eq. '9') then    ! new set of runs
         seed = make_sprng_seed() ! make seed from date/time information 
         junkPtr = init_sprng(seed, CRAYLCG) ! initialize stream   
         junk = print_sprng()
         in_old = 0
         n_old = 0
      else                           ! continue from previously stored state
         open(31, file = filename, status = 'unknown', 
     &      form = 'unformatted')
         read(31) in_old, n_old, size, buffer  !read previous run info.
         junkPtr = unpack_sprng(buffer)
         close (31)
      endif

      return 
      end

C-- save the final state of the default stream into a file
      subroutine save_state(filename, in, n)
      implicit none
#include "sprng_f.h"

      integer in, n, size
      character filename*7, buffer(MAX_PACKED_LENGTH)

      open(31, file = filename, status = 'unknown', 
     &      form = 'unformatted')

      size = pack_sprng(buffer)
      write(31)in,n,size,buffer  !write the current run info. for future use
      close(31)

      return 
      end
