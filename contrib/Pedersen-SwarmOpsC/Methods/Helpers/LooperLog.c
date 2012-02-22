/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	LooperLog
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/Helpers/LooperLog.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Matrix.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

struct SO_LooperLogContext SO_MakeLooperLogContext(SO_FProblem f, struct SO_MethodContext *fContext, size_t numRuns)
{
	struct SO_LooperLogContext c;

	c.f = f;
	c.fContext = fContext;
	c.numRuns = numRuns;
	c.numExecuted = 0;

	c.results = SO_NewMatrix(numRuns, fContext->fDim);
	c.fitnessResults = SO_NewFitnessVector(numRuns);

	return c;
}

/* ---------------------------------------------------------------- */

void SO_FreeLooperLogContext(struct SO_LooperLogContext *c)
{
	SO_FreeMatrix(c->results, c->numRuns);
	c->results = 0;

	SO_FreeVector(c->fitnessResults);
	c->fitnessResults = 0;
}

/* ---------------------------------------------------------------- */

SO_TFitness SO_LooperLog(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit)
{
	/* Cast void-ptr context to correct struct-type. */
	struct SO_LooperLogContext* c = (struct SO_LooperLogContext*) context;

	/* Clone context to local variables for easier reference. */
	SO_FProblem f = c->f;
	struct SO_MethodContext *fContext = c->fContext;
	size_t numRuns = c->numRuns;

	SO_TElm **results = c->results;
	SO_TFitness *fitnessResults = c->fitnessResults;

	SO_TFitness fitnessSum = 0;

	size_t i;

	for (i=0; i<numRuns && fitnessSum<fitnessLimit; i++)
	{
		SO_TFitness fitness = f(param, fContext, fitnessLimit-fitnessSum);

		/* Ensure fitness is non-negative so Pre-Emptive Fitness Evaluation can be used. */
		assert(fitness>=0);

		fitnessSum += fitness;

		SO_CopyVector(results[i], fContext->g, fContext->fDim);
		fitnessResults[i] = fContext->gFitness;
	}

	c->numExecuted = i;

	return fitnessSum;
}

/* ---------------------------------------------------------------- */
