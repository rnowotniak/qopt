/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	HC
 *
 *	See header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/HC.h>
#include <SwarmOps/Tools/Bound.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Init.h>
#include <SwarmOps/Tools/Random.h>
#include <math.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

/* The method has the following parameters:
 * Index 0: Sampling range
 * Index 1: Probability weight
 */

const char SO_kNameHC[] = "HC";
const char* SO_kParameterNameHC[] = {"r", "D"};

const SO_TElm SO_kParametersDefaultHC[SO_kNumParametersHC] = {0.01, 10.0};
const SO_TElm SO_kParametersLowerHC[SO_kNumParametersHC] = {0.0000001, 0.001};
const SO_TElm SO_kParametersUpperHC[SO_kNumParametersHC] = {1, 10000};

/* ---------------------------------------------------------------- */

SO_TElm SO_HCRange(const SO_TElm *param)
{
	assert(param);

	return param[0];
}

SO_TElm SO_HCWeight(const SO_TElm *param)
{
	assert(param);

	return param[1];
}

/* ---------------------------------------------------------------- */

/* Return the probability of movement computed from fitnesses of
 * current and new potential position. */
SO_TFitness SO_HCProbability(SO_TFitness fitnessX, SO_TFitness fitnessY, SO_TElm d)
{
	return 1.0/(1+exp((fitnessY-fitnessX)/d));
}

/* ---------------------------------------------------------------- */

/* The overall structure of this function is:
 * - Retrieve variables from context and parameters.
 * - Allocate and initialize vectors and other data needed by this optimization method.
 * - Perform optimization.
 * - De-allocate data.
 * - Return best result from the optimization.
 */

SO_TFitness SO_HC(const SO_TElm *param, void* context, const SO_TFitness fitnessLimit)
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
	SO_TElm *g = c->g;								/* Best found position for this run. */
	SO_TFitness *gFitness = &(c->gFitness);			/* Fitness for best found position. */

	/* Retrieve parameter specific to HC method. */
	SO_TElm r = SO_HCRange(param);
	SO_TElm D = SO_HCWeight(param);

	/* Allocate agent position and search-range. */
	SO_TElm *x = SO_NewVector(n);
	SO_TElm *y = SO_NewVector(n);
	SO_TElm *d = SO_NewVector(n);

	/* Iteration variables. */
	size_t i, j;

	/* Fitness variables. */
	SO_TFitness fitness, newFitness;

	/* Initialize search-range to full search-space. */
	SO_InitRange(d, n, lowerBound, upperBound);

	/* Scale search-range. */
	for (j=0; j<n; j++)
	{
		d[j] *= r;
	}

	/* Initialize agent-position in search-space. */
	SO_InitUniform(x, n, lowerInit, upperInit);

	/* Compute fitness of initial position.
	 * This counts as an iteration below. */
	*gFitness = fitness = f(x, fContext, SO_kFitnessMax);

	/* Trace fitness of best found solution. */
	SO_SetFitnessTrace(c, 0, *gFitness);

	for (i=1; i<numIterations; i++)
	{
		/* Compute potentially new position. */
		for (j=0; j<n; j++)
		{
			y[j] = x[j] + d[j] * SO_RandBi();
		}

		/* Enforce bounds before computing new fitness. */
		SO_BoundAll(y, n, lowerBound, upperBound);

		/* Compute new fitness.
		 * Do not use pre-emptive fitness evaluation. */
		newFitness = f(y, fContext, SO_kFitnessMax);

		/* Update best-known fitness and position.
		 * Since movement to worse position may occur,
		 * this must be done on every move to ensure
		 * a good position is not lost. */
		if (newFitness<*gFitness)
		{
			/* Update this run's best known position. */
			SO_CopyVector(g, y, n);

			/* Update this run's best known fitness. */
			*gFitness = newFitness;
		}

		/* Update position. */
		if (SO_RandUni() < SO_HCProbability(fitness, newFitness, D))
		{
			fitness = newFitness;

			/* We could just swap pointers between x and y here,
			 * but that might not work with gbest to be implemented lateron. */
			for (j=0; j<n; j++)
			{
				x[j] = y[j];
			}
		}

		/* Trace fitness of best found solution. */
		SO_SetFitnessTrace(c, i, *gFitness);
	}

	/* Update all-time best known position. */
	SO_MethodUpdateBest(c, g, *gFitness);

	/* Delete agent's position and search-range vectors. */
	SO_FreeVector(x);
	SO_FreeVector(y);
	SO_FreeVector(d);

	return *gFitness;
}

/* ---------------------------------------------------------------- */
