#include <cstdio>
#include <cstdlib>
#include <exception>
#include "sat.h"


SAT::SAT(const char *fname) {
	FILE *f = fopen(fname, "r");
	if (!f) {
		perror("Error");
		return;
	}
	initprob(f);
	fclose(f);
	printf("%d %d\n", numatom, numclause);
	printf("%d\n", clause[3][2]);
}

float SAT::evaluator(char *cand, int length) {
	int true_clauses = 0;
	for (int ci = 0; ci < numclause; ci++) {
		int clause_is_true = 0;
		for (int li = 0; li < size[ci]; li++) {
			int idx = abs(clause[ci][li]) - 1;
			int state = cand[idx] != '0' ? 1 : 0;
			if (clause[ci][li] < 0) {
				state = !state;
			}
			clause_is_true |= state;
			// if (clause_is_true) { // optimization
			// 	break;
			// }
		}
		if (clause_is_true) {
			true_clauses++;
		}
		// printf("%d ",clause_is_true);
		// printf("\n");
	}
	return true_clauses;
}

class SomeException : public std::exception {

	const char *str;

	public:

	SomeException(const char *str) : str(str) { }

	virtual const char *what() const throw()
	{
		return str;
	}
};

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
		throw SomeException("Bad input file");
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
	//int res = s1.evaluator("111101101101101101101101101101101101101101101101101101101101101101101101101101101101101101");
	//int res = s1.evaluator("100100001100100100001100010100010001010010100010100010010010010010001001001100001001001001");
	//printf("res: %d\n", res);
}

