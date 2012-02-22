/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	SA
 *
 *	See header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/SA.h>
#include <SwarmOps/Tools/Bound.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Init.h>
#include <SwarmOps/Tools/Random.h>
#include <math.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

/* The method has the following parameters:
 * Index 0: r, sampling range factor
 * Index 1: alpha, start-value
 * Index 2: beta, end-value
 * Index 3: T, iterations between resets.
 */

const char SO_kNameSA[] = "SA";
const char* SO_kParameterNameSA[] = {"r", "alpha", "beta", "T"};

const SO_TElm SO_kParametersDefaultSA[SO_kNumParametersSA] = {0.01, 0.3, 0.01, 40000};
const SO_TElm SO_kParametersLowerSA[SO_kNumParametersSA] = {1e-5, 1e-5, 1e-5, 100};
const SO_TElm SO_kParametersUpperSA[SO_kNumParametersSA] = {1, 1, 1, 100000};

/* ---------------------------------------------------------------- */

SO_TElm SO_SARange(const SO_TElm *param)
{
	assert(param);

	return param[0];
}

SO_TElm SO_SAAlpha(const SO_TElm *param)
{
	assert(param);

	return param[1];
}

SO_TElm SO_SABeta(const SO_TElm *param)
{
	assert(param);

	return param[2];
}

size_t SO_SATime(const SO_TElm *param)
{
	assert(param);

	return (size_t) (param[3]+0.5);
}

/* ---------------------------------------------------------------- */

/* Return the probability of movement computed from fitnesses of
 * current and new potential position. */
SO_TFitness SO_SAProbability(SO_TFitness fitnessX, SO_TFitness fitnessY, SO_TElm d)
{
	return exp((fitnessX-fitnessY)/d);
}

/* ---------------------------------------------------------------- */

/* The overall structure of this function is:
 * - Retrieve variables from context and parameters.
 * - Allocate and initialize vectors and other data needed by this optimization method.
 * - Perform optimization.
 * - De-allocate data.
 * - Return best result from the optimization.
 */

SO_TFitness SO_SA(const SO_TElm *param, void* context, const SO_TFitness fitnessLimit)
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

	/* Retrieve parameter specific to SA method. */
	SO_TElm r = SO_SARange(param);
	SO_TElm alpha = SO_SAAlpha(param);
	SO_TElm beta = SO_SABeta(param);
	size_t T = SO_SATime(param);

	/* Allocate agent position and search-range. */
	SO_TElm *x = SO_NewVector(n);
	SO_TElm *y = SO_NewVector(n);
	SO_TElm *d = SO_NewVector(n);

	/* Probability weight and its exponential decrease factor. */
	SO_TElm D;
	SO_TElm gamma = pow(beta/alpha, 1.0/T);

	/* Iteration variables. */
	size_t i, j, t;

	/* Fitness variables. */
	SO_TFitness fitness, newFitness;

	/* Initialize best-found fitness. */
	*gFitness = SO_kFitnessMax;

	/* Initialize search-range to full search-space. */
	SO_InitRange(d, n, lowerBound, upperBound);

	/* Scale search-range. */
	for (j=0; j<n; j++)
	{
		d[j] *= r;
	}

	for (i=0, t = T; i<numIterations; i++, t++)
	{
		/* Reset x and D for every T steps. */
		if (t == T)
		{
			/* Reset time-step counter. */
			t = 0;

			/* Reset probability-weight. */
			D = alpha;

			/* Initialize agent-position in search-space. */
			SO_InitUniform(x, n, lowerInit, upperInit);

			/* Compute fitness of new position.
			 * Do not use pre-emptive fitness evaluation. */
			fitness = f(x, fContext, SO_kFitnessMax);

			if (fitness<*gFitness)
			{
				/* Update this run's best known position. */
				SO_CopyVector(g, x, n);

				/* Update this run's best know fitness. */
				*gFitness = fitness;
			}

			/* Trace fitness of best found solution. */
			SO_SetFitnessTrace(c, i, *gFitness);

			/* This counts as an iteration. */
			i++;
			t++;
		}

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

			/* Update this run's best know fitness. */
			*gFitness = newFitness;
		}

		/* Update position. */
		if ((newFitness<fitness) || (SO_RandUni() < SO_SAProbability(fitness, newFitness, D)) )
		{
			fitness = newFitness;

			/* We could just swap pointers between x and y here,
			 * but that might not work with gbest to be implemented lateron. */
			for (j=0; j<n; j++)
			{
				x[j] = y[j];
			}
		}

		/* Decrease probability weight exponentially. */
		D *= gamma;

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
