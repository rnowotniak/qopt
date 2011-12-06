/****************************************************************************/
/*                ____Demonstrates checkpointing____                        */
/* In a new run, this program initializes a random number stream and prints */
/* a few random numbers. Finally it packs the state of the stream into a    */
/* file. In a continuation of an old run this program reads in the state of */
/* a stream from a file, prints a few random numbers, and stores the final  */
/* state of the stream into a file.                                         */
/****************************************************************************/


#include <stdio.h>

#define SIMPLE_SPRNG		/* simple interface                         */
#include "sprng.h"              /* SPRNG header file                        */

#define SEED 985456376



main(int argc, char *argv[])
{
  int i, size;
  double rn;
  FILE *fp;
  char buffer[MAX_PACKED_LENGTH], outfile[80], infile[80], *bytes;
  
  
  /*********************** Initialize streams *******************************/
       
  printf("Enter name of file to store final state of the stream:\n");
  scanf("%s", outfile);
  printf("Enter name of file to read from:\n\t(enter 9 for a new run)\n");
  scanf("%s", infile);
  
  if(infile[0] == '9')		/* initialize stream the first time         */
    init_sprng(SEED,SPRNG_DEFAULT);
  else           		/* read stream state from file afterwards   */
  {
    fp = fopen(infile,"r");
    if(fp == NULL)
    {
      fprintf(stderr,"Could not open file %s for reading\nCheck path or permissions\n", infile);
      exit(1);
    }
    fread(&size,1,sizeof(int),fp);
    fread(buffer,1,size,fp);
    unpack_sprng(buffer);
    fclose(fp);
  }
  
  /*********************** print random numbers *****************************/
            
  printf(" Printing 5 random numbers in [0,1): \n");
  
  for(i=0; i<5; i++)
  {
    rn = sprng();	       /* generate a double precision random number */
    printf("%d  %f\n", i+1, rn);
  }
  
  /************************* store stream state *****************************/
            
  size = pack_sprng(&bytes);   /* pack stream state into an array           */
  fp = fopen(outfile,"w");     /* open file to store stream state           */
  if(fp == NULL)
  {
    fprintf(stderr,"Could not open file %s for writing\nCheck path or permissions\n", outfile);
    exit(1);
  }
  fwrite(&size,1,sizeof(int),fp); /* store # of bytes required for storage  */
  fwrite(bytes,1,size,fp);     /* store stream state                        */
  fclose(fp);

  free(bytes);		       /* free memory needed to store stream state  */

}
