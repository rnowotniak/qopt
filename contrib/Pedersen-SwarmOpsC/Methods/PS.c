/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	PS
 *
 *	See header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/PS.h>
#include <SwarmOps/Tools/Bound.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Init.h>
#include <SwarmOps/Tools/Random.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

/* The PS method has no parameters. */

const char SO_kNamePS[] = "PS";

/* ---------------------------------------------------------------- */

/* The overall structure of this function is:
 * - Retrieve variables from context and parameters.
 * - Allocate and initialize vectors and other data needed by this optimization method.
 * - Perform optimization.
 * - De-allocate data.
 * - Return best result from the optimization.
 */

SO_TFitness SO_PS(const SO_TElm *param, void* context, const SO_TFitness fitnessLimit)
{
	/* Cast void-ptr context to correct struct-type. */
	struct SO_MethodContext *c = (struct SO_MethodContext*) context;

	/* Clone context to local variables for easier reference. */
	SO_FProblem f = c->f;							/* Fitness function. */
	void *fContext = c->fContext;					/* Context for fitness function. */
	SO_TDim n = c->fDim;							/* Dimensionality of problem. */
	SO_TElm const* lowerInit = c->lowerInit;		/* Lower initialization boundary. */
	SO_TElm const* upperInit = c->upperInit;		/* Upper initialization boundary. */
	SO_TElm const* lowerBound = c->lowerBound;		/* Lower search-space boundary. */
	SO_TElm const* upperBound = c->upperBound;		/* Upper search-space boundary. */
	size_t numIterations = c->numIterations;		/* Number of iterations to perform. */

	/* Allocate agent position and search-range. */
	SO_TElm* x = SO_NewVector(n);
	SO_TElm* d = SO_NewVector(n);

	/* Iteration counter. */
	size_t i;

	/* Fitness variables. */
	SO_TFitness fitness, newFitness;

	/* Initialize agent-position in search-space. */
	SO_InitUniform(x, n, lowerInit, upperInit);

	/* Initialize search-range to full search-space. */
	SO_InitRange(d, n, lowerBound, upperBound);

	/* Compute fitness of initial position.
	 * This counts as an iteration below. */
	fitness = f(x, fContext, SO_kFitnessMax);

	/* Trace fitness of best found solution. */
	SO_SetFitnessTrace(c, 0, fitness);

	for (i=1; i<numIterations; i++)
	{
		/* Pick random dimension. */
		SO_TDim R = SO_RandIndex(n);

		/* Store old value for that dimension. */
		SO_TElm t = x[R];

		/* Compute new value for that dimension. */
		x[R] += d[R];

		/* Enforce bounds before computing new fitness. */
		SO_BoundOne(x, R, lowerBound, upperBound);

		/* Compute new fitness. */
		newFitness = f(x, fContext, fitness);

		/* If improvement to fitness, keep new position. */
		if (newFitness < fitness)
		{
			fitness = newFitness;
		}
		/* Otherwise restore position, and reduce and invert search-range. */
		else
		{
			x[R] = t;
			d[R] *= -0.5;
		}

		/* Trace fitness of best found solution. */
		SO_SetFitnessTrace(c, i, fitness);
	}

	/* Set best position found in this run. */
	SO_MethodSetResult(c, x, fitness);

	/* Update all-time best known position. */
	SO_MethodUpdateBest(c, x, fitness);

	/* Delete agent's position and search-range vectors. */
	SO_FreeVector(x);
	SO_FreeVector(d);

	return fitness;
}

/* ---------------------------------------------------------------- */
