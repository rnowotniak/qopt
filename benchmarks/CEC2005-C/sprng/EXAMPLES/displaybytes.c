#include <stdio.h>


main()
{
  int c;
  
  while( (c=getchar())!=EOF )
    printf("%x ",c);
  
  putchar('\n');
  

}
