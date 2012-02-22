/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	LUS
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/LUS.h>
#include <SwarmOps/Tools/Bound.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Init.h>
#include <SwarmOps/Tools/Random.h>
#include <SwarmOps/Tools/Sample.h>
#include <math.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

/* The LUS method has the following parameters:
 * Index 0: gamma
 */

const char SO_kNameLUS[] = "LUS";
const char* SO_kParameterNameLUS[] = {"gamma"};

const SO_TElm SO_kParametersDefaultLUS[SO_kNumParametersLUS] = {3.0};
const SO_TElm SO_kParametersLowerLUS[SO_kNumParametersLUS] = {0.5};
const SO_TElm SO_kParametersUpperLUS[SO_kNumParametersLUS] = {20.0};

#if 0
const SO_TElm SO_kParametersDefaultLUS[SO_kNumParametersLUS] = {5.311354}; /* Meta-optimization with 20x iterations */
const SO_TElm SO_kParametersDefaultLUS[SO_kNumParametersLUS] = {7.572575}; /* Meta-optimization with 30x iterations */
const SO_TElm SO_kParametersDefaultLUS[SO_kNumParametersLUS] = {8.366221}; /* Meta-optimization with 40x iterations */
#endif

/* ---------------------------------------------------------------- */

/* Note that gamma is the reciprocal of alpha (or sometimes, beta)
 * in the literature. This gives a better distribution in
 * meta-optimization. */

SO_TElm SO_LUSGamma(const SO_TElm *param)
{
	assert(param);

	return param[0];
}

/* ---------------------------------------------------------------- */

SO_TElm SO_LUSSample(const SO_TElm x, const SO_TElm range, const SO_TElm lowerBound, const SO_TElm upperBound)
{
#if 1
	return SO_SampleBoundedOne(x, range, lowerBound, upperBound);
#else
	/* First sample from full range, and then bound the sample.
	 * This causes many samples to be boundary points, which is
	 * generally undesirable. */

	SO_TElm y = x + range * SO_RandBi();

	y = SO_Bound(y, lowerBound, upperBound);

	return y;
#endif
}

/* ---------------------------------------------------------------- */

/* The overall structure of this function is:
 * - Retrieve variables from context and parameters.
 * - Allocate and initialize vectors and other data needed by this optimization method.
 * - Perform optimization.
 * - De-allocate data.
 * - Return best result from the optimization.
 */

SO_TFitness SO_LUS(const SO_TElm *param, void* context, const SO_TFitness fitnessLimit)
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

	/* Retrieve parameter specific to LUS method. */
	SO_TElm gamma = SO_LUSGamma(param);				/* gamma = 1.0/alpha */

	/* Allocate agent position and search-range vectors. */
	SO_TElm *x = SO_NewVector(n);					/* Current position. */
	SO_TElm *y = SO_NewVector(n);					/* Potentially new position. */
	SO_TElm *d = SO_NewVector(n);					/* Search-range. */

	/* Initialize search-range and decrease-factor. */
	SO_TElm r = 1;									/* Search-range. */
	SO_TElm q = pow(2.0, -1.0/(n*gamma));			/* Decrease-factor (using gamma = 1.0/alpha). */
	/* SO_TElm q = pow(2.0, -alpha/n); */			/* Decrease-factor (using alpha). */

	/* Iteration variables. */
	size_t i, j;

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
		/* Compute potentially new position. */
		for (j=0; j<n; j++)
		{
			/* Pick a sample from the neighbourhood of the current
			 * position and within the given range. The sample
			 * is assumed bounded by the SO_LUSSample() function. */
			y[j] = SO_LUSSample(x[j], r * d[j], lowerBound[j], upperBound[j]);
		}

		/* Compute new fitness. */
		newFitness = f(y, fContext, fitness);

		/* Update position and fitness in case of strict improvement. */
		if (newFitness < fitness)
		{
			/* Update fitness. */
			fitness = newFitness;

			/* Update position.
			 * We could just swap pointers between x and y here, but
			 * that might not work with parallel implementation later on. */
			SO_CopyVector(x, y, n);
		}
		else
		{
			/* Decrease the search-range. */
			r *= q;
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
	SO_FreeVector(y);
	SO_FreeVector(d);

	/* Return best-found fitness. */
	return fitness;
}

/* ---------------------------------------------------------------- */
