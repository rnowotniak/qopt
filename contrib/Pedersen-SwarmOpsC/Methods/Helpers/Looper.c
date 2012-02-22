/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Looper
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/Helpers/Looper.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

struct SO_LooperContext SO_MakeLooperContext(SO_FProblem f, void *fContext, size_t numRuns)
{
	struct SO_LooperContext c;

	c.f = f;
	c.fContext = fContext;
	c.numRuns = numRuns;

	return c;
}

/* ---------------------------------------------------------------- */

void SO_FreeLooperContext(struct SO_LooperContext *c)
{
	/* Do nothing. */
}

/* ---------------------------------------------------------------- */

SO_TFitness SO_Looper(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit)
{
	/* Cast void-ptr context to correct struct-type. */
	struct SO_LooperContext* c = (struct SO_LooperContext*) context;

	/* Clone context to local variables for easier reference. */
	SO_FProblem f = c->f;
	void *fContext = c->fContext;
	size_t numRuns = c->numRuns;

	SO_TFitness fitnessSum = 0;

	size_t i;

	for (i=0; i<numRuns && fitnessSum<fitnessLimit; i++)
	{
		SO_TFitness fitness = f(param, fContext, fitnessLimit-fitnessSum);

		/* Ensure fitness is non-negative so Pre-Emptive Fitness Evaluation can be used. */
		assert(fitness>=0);

		fitnessSum += fitness;
	}

	c->numExececuted = i;

	return fitnessSum;
}

/* ---------------------------------------------------------------- */
