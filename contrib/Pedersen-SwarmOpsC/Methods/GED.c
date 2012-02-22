/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	GED
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/GED.h>
#include <SwarmOps/Tools/Bound.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Init.h>
#include <SwarmOps/Tools/Random.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

/* ---------------------------------------------------------------- */

/* The GED method has the following parameters:
 * Index 0: alpha (gradient stepsize)
 * Index 1: probability p
 */

const char SO_kNameGED[] = "GED";
const char* SO_kParameterNameGED[] = {"alpha", "p"};

const SO_TElm SO_kParametersDefaultGED[SO_kNumParametersGED] = {0.05, 0.05};
const SO_TElm SO_kParametersLowerGED[SO_kNumParametersGED] = {0, 0};
const SO_TElm SO_kParametersUpperGED[SO_kNumParametersGED] = {2, 1};

/* ---------------------------------------------------------------- */

SO_TElm SO_GEDAlpha(const SO_TElm *param)
{
	assert(param);

	return param[0];
}

SO_TElm SO_GEDP(const SO_TElm *param)
{
	assert(param);

	return param[1];
}

/* ---------------------------------------------------------------- */

/* The overall structure of this function is:
 * - Retrieve variables from context and parameters.
 * - Allocate and initialize vectors and other data needed by this optimization method.
 * - Perform optimization.
 * - De-allocate data.
 * - Return best result from the optimization.
 */

SO_TFitness SO_GED(const SO_TElm *param, void* context, const SO_TFitness fitnessLimit)
{
	/* Cast void-ptr context to correct struct-type. */
	struct SO_MethodContext *c = (struct SO_MethodContext*) context;

	/* Clone context to local variables for easier reference. */
	SO_FProblem f = c->f;							/* Fitness function. */
	SO_FGradient fGradient = c->fGradient;			/* Gradient for fitness function. */
	void *fContext = c->fContext;					/* Context for fitness function. */
	SO_TDim n = c->fDim;							/* Dimensionality of problem. */
	SO_TElm const* lowerInit = c->lowerInit;		/* Lower initialization boundary. */
	SO_TElm const* upperInit = c->upperInit;		/* Upper initialization boundary. */
	SO_TElm const* lowerBound = c->lowerBound;		/* Lower search-space boundary. */
	SO_TElm const* upperBound = c->upperBound;		/* Upper search-space boundary. */
	size_t numIterations = c->numIterations;		/* Number of iterations to perform. */
	SO_TElm *g = c->g;								/* Best found position for this run. */
	SO_TFitness *gFitness = &(c->gFitness);			/* Fitness for best found position. */

	/* Retrieve parameter specific to GED method. */
	SO_TElm alpha = SO_GEDAlpha(param);				/* Step-size. */
	SO_TElm p = SO_GEDP(param);						/* Fitness evaluation probability. */

	/* Allocate agent position and search-range. */
	SO_TElm *x = SO_NewVector(n);					/* Current position. */
	SO_TElm *v = SO_NewVector(n);					/* Gradient/velocity. */

	/* Gradient variables. */
	SO_TElm gradientNorm, stepsize;

	/* Iteration variables. */
	size_t i, j;

	/* Fitness variables. */
	SO_TFitness fitness;

	assert(fGradient);

	/* Initialize agent-position in search-space. */
	SO_InitUniform(x, n, lowerInit, upperInit);

	/* Compute fitness of initial position.
	 * This counts as an iteration below. */
	fitness = f(x, fContext, SO_kFitnessMax);
	SO_CopyVector(g, x, n);
	*gFitness = fitness;

	/* Trace fitness of best found solution. */
	SO_SetFitnessTrace(c, 0, *gFitness);

	for (i=1; i<numIterations; i++)
	{
		/* Compute gradient. */
		i += fGradient(x, v, fContext);

		/* Compute norm of gradient-vector. */
		gradientNorm = SO_Norm(v, n);

		/* Compute current stepsize. */
		stepsize = alpha / gradientNorm;

		/* Move in direction of gradient. */
		for (j=0; j<n; j++)
		{
			x[j] -= stepsize * v[j];
		}

		/* Enforce bounds before computing new fitness. */
		SO_BoundAll(x, n, lowerBound, upperBound);

		if (SO_RandUni()<p)
		{
			/* Compute new fitness. */
			fitness = f(x, fContext, *gFitness);

			/* Update best fitness found in this run. */
			if (fitness<*gFitness)
			{
				/* Update this run's best known position. */
				SO_CopyVector(g, x, n);

				/* Update this run's best know fitness. */
				*gFitness = fitness;
			}
		}

		/* Trace fitness of best found solution. */
		SO_SetFitnessTrace(c, i, *gFitness);
	}

	/* Update all-time best known position. */
	SO_MethodUpdateBest(c, g, *gFitness);

	/* Delete agent's position and search-range vectors. */
	SO_FreeVector(x);
	SO_FreeVector(v);

	/* Return best-found fitness. */
	return *gFitness;
}

/* ---------------------------------------------------------------- */
