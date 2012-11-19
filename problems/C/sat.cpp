#include <cstdio>
#include <cstdlib>
#include "sat.h"


SAT::SAT(const char *fname) {
	FILE *f = fopen(fname, "r");
	initprob(f);
	fclose(f);
	printf("%d %d\n", numatom, numclause);
	printf("%d\n", clause[3][2]);
}

float SAT::evaluate(const char *cand, int length) {
	int result = 0;
	for (int ci = 0; ci < numclause; ci++) {
		int foo = 0;
		for (int li = 0; li < size[ci]; li++) {
			int idx = abs(clause[ci][li]) - 1;
			int state = cand[idx] != '0' ? 1 : 0;
			if (clause[ci][li] < 0) {
				state = !state;
			}
			foo |= state;
		}
		if (foo) {
			result++;
		}
		// printf("%d ",foo);
		// printf("\n");
	}
	return result;
}

void SAT::initprob(FILE *F) // drawn from WalkSAT
{
	int i;			/* loop counter */
	int j;			/* another loop counter */
	int lastc;
	int nextc;
	int *storeptr = 0;
	int freestore;
	int lit;

	while ((lastc = getc(F)) == 'c')
	{
		while ((nextc = getc(F)) != EOF && nextc != '\n');
	}
	ungetc(lastc,F);
	if (fscanf(F,"p cnf %i %i",&numatom,&numclause) != 2)
	{
		fprintf(stderr,"Bad input file\n");
		exit(-1);
	}
	if(numatom > MAXATOM)
	{
		fprintf(stderr,"ERROR - too many atoms\n");
		exit(-1);
	}

#ifdef Huge
	clause = (int **) malloc(sizeof(int *)*(numclause+1));
	size = (int *) malloc(sizeof(int)*(numclause+1));
	false = (int *) malloc(sizeof(int)*(numclause+1));
	lowfalse = (int *) malloc(sizeof(int)*(numclause+1));
	wherefalse = (int *) malloc(sizeof(int)*(numclause+1));
	numtruelit = (int *) malloc(sizeof(int)*(numclause+1));
#else
	if(numclause > MAXCLAUSE)                     
	{                                      
		fprintf(stderr,"ERROR - too many clauses\n"); 
		exit(-1);                              
	}                                        
#endif
	freestore = 0;
	numliterals = 0;
	for(i = 0;i < 2*MAXATOM+1;i++)
		numoccurence[i] = 0;
	for(i = 0;i < numclause;i++)
	{
		size[i] = -1;
		if (freestore < MAXLENGTH)
		{
			storeptr = (int *) malloc( sizeof(int) * STOREBLOCK );
			freestore = STOREBLOCK;
			fprintf(stderr,"allocating memory...\n");
		}
		clause[i] = storeptr;
		do
		{
			size[i]++;
			if(size[i] > MAXLENGTH)
			{
				printf("ERROR - clause too long\n");
				exit(-1);
			}
			if (fscanf(F,"%i ",&lit) != 1)
			{
				fprintf(stderr, "Bad input file\n");
				exit(-1);
			}
			if(lit != 0)
			{
				*(storeptr++) = lit; /* clause[i][size[i]] = j; */
				freestore--;
				numliterals++;
				numoccurence[lit+MAXATOM]++;
			}
		}
		while(lit != 0);
	}
	if(size[0] == 0)
	{
		fprintf(stderr,"ERROR - incorrect problem format or extraneous characters\n");
		exit(-1);
	}

	for(i = 0;i < 2*MAXATOM+1;i++)
	{
		if (freestore < numoccurence[i])
		{
			storeptr = (int *) malloc( sizeof(int) * STOREBLOCK );
			freestore = STOREBLOCK;
			fprintf(stderr,"allocating memory...\n");
		}
		occurence[i] = storeptr;
		freestore -= numoccurence[i];
		storeptr += numoccurence[i];
		numoccurence[i] = 0;
	}

	for(i = 0;i < numclause;i++)
	{
		for(j = 0;j < size[i];j++)
		{
			occurence[clause[i][j]+MAXATOM]
				[numoccurence[clause[i][j]+MAXATOM]] = i;
			numoccurence[clause[i][j]+MAXATOM]++;
		}
	}
}
int main() {
	SAT s1 = SAT("../sat/flat30-100.cnf");
	//int res = s1.evaluate("111101101101101101101101101101101101101101101101101101101101101101101101101101101101101101");
	//int res = s1.evaluate("100100001100100100001100010100010001010010100010100010010010010010001001001100001001001001");
	//printf("res: %d\n", res);
}

