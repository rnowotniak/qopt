
C
C        ____Demonstrates the use of sprng without initialization____      */
C A few double precision random numbers are generated, without the user    */
C explicitly initializing the streams. SPRNG automatically initializes a   */
C stream with default seed and parameter the first time 'sprng' is called. */
C

       program simplef_simple
       implicit none

#define SIMPLE_SPRNG	
#include "sprng_f.h"

       integer i
       real*8 rn
       
       print *, 'Printing 3 random numbers in [0,1):'
       do 100 i = 1, 3
          rn = sprng()
          write(*,"(f8.6)")  rn
 100   continue


       end
