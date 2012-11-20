#ifndef _SAT_H
#define _SAT_H 1

#include "framework.h"

#define MAXATOM 100000		/* maximum possible number of atoms */
#define MAXCLAUSE 500000	/* maximum possible number of clauses */
#define MAXLENGTH 500           /* maximum number of literals which can be in any clause */
#define STOREBLOCK 2000000	/* size of block to malloc each time */

class SAT : public Problem {

	void initprob(FILE *F);

	int numatom;
	int numclause;
	int numliterals;

	int size[MAXCLAUSE];		/* length of each clause */
	int numoccurence[2*MAXATOM+1];	/* number of times each literal occurs */
	int * clause[MAXCLAUSE];	/* clauses to be satisfied */
	int *occurence[2*MAXATOM+1];	/* where each literal occurs */

	public:

		SAT(const char *fname);

		virtual float evaluator (char *x, int length);

		virtual void repairer (char *x, int length) { }

		virtual long double r_evaluator(long double *x, int length) {
			return -1;
		}

};

#endif
