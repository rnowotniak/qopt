/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Multi
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/Helpers/Multi.h>
#include <SwarmOps/Tools/Memory.h>
#include <stdlib.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

/* Struct used in sorting the multiple problems. */

struct SO_MultiIndex
{
	SO_TFitness fitness;		/* Last fitness obtained. This is sorting key. */
	size_t index;				/* The problem's index number. */
};

/* ---------------------------------------------------------------- */

struct SO_MultiContext SO_MakeMultiContext(SO_FMethod method, void **contexts, size_t numProblems)
{
	struct SO_MultiContext c;
	size_t i;

	c.method = method;
	c.contexts = contexts;
	c.numProblems = numProblems;
	c.ordering = (struct SO_MultiIndex*) SO_MAlloc(sizeof(struct SO_MultiIndex) * numProblems);

	/* Initialize ordering of problems. */
	for (i=0; i<numProblems; i++)
	{
		c.ordering[i].index = i;
	}

	return c;
}

/* ---------------------------------------------------------------- */

void SO_FreeMultiContext(struct SO_MultiContext *c)
{
	free(c->ordering);
	c->ordering = 0;
}

/* ---------------------------------------------------------------- */

/* Compare the fitness values of two problems. Used in sorting.
 * Return -1 if f(a) >  f(b)
 * Return  0 if f(a) == f(b) 
 * Return  1 if f(a) <  f(b)
 */

int SO_MultiIndexCompare(const void *a, const void *b)
{
	const struct SO_MultiIndex *l = (const struct SO_MultiIndex*) a;
	const struct SO_MultiIndex *r = (const struct SO_MultiIndex*) b;

	SO_TFitness diff = (l->fitness - r->fitness);

	int retVal;

	if (diff>0)
	{
		retVal = -1;
	}
	else if (diff<0)
	{
		retVal = 1;
	}
	else
	{
		retVal = 0;
	}

	return retVal;
}

/* ---------------------------------------------------------------- */

SO_TFitness SO_Multi(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit)
{
	/* Cast void-ptr context to correct struct-type. */
	struct SO_MultiContext const* c = (struct SO_MultiContext const*) context;

	/* Clone context to local variables for easier reference. */
	SO_FMethod method = c->method;
	void **contexts = c->contexts;
	size_t numProblems = c->numProblems ;
	struct SO_MultiIndex* ordering = c->ordering;

	SO_TFitness fitnessSum = 0;

	size_t i;

	for (i=0; i<numProblems && fitnessSum<fitnessLimit; i++)
	{
		size_t contextIndex = ordering[i].index;
		SO_TFitness fitness = method(param, contexts[contextIndex], fitnessLimit-fitnessSum);

		/* Ensure fitness is non-negative so Pre-Emptive Fitness Evaluation can be used. */
		assert(fitness>=0);

		ordering[i].fitness = fitness;

		fitnessSum += fitness;
	}

	/* Sort the evaluation order of the problems for use in next call to SO_Multi. */
	qsort(ordering, numProblems, sizeof(struct SO_MultiIndex), SO_MultiIndexCompare);

	return fitnessSum;
}

/* ---------------------------------------------------------------- */
