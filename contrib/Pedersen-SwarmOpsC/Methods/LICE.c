/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	LICE
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/LICE.h>
#include <SwarmOps/Tools/Bound.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Init.h>
#include <SwarmOps/Tools/Random.h>
#include <SwarmOps/Tools/Sample.h>
#include <math.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

/* The LICE method has the following parameters:
 * Index 0: gamma2
 * Index 1: N
 * Index 2: gamma
 */

const char SO_kNameLICE[] = "LICE";
const char* SO_kParameterNameLICE[] = {"gamma_2", "N", "gamma"};

#define SO_kLICEGammaLower 0.5
#define SO_kLICEGammaUpper 6.0

const SO_TElm SO_kParametersDefaultLICE[SO_kNumParametersLICE] = {0.991083, 25.0, 5.633202}; /* Meta-optimized for ANN Cancer problem. */
const SO_TElm SO_kParametersLowerLICE[SO_kNumParametersLICE] = {0.5, 10.0, SO_kLICEGammaLower};
const SO_TElm SO_kParametersUpperLICE[SO_kNumParametersLICE] = {4.0, 40.0, SO_kLICEGammaUpper};

/* ---------------------------------------------------------------- */

SO_TElm SO_LICEGamma2(const SO_TElm *param)
{
	assert(param);

	return param[0];
}

size_t SO_LICEN(SO_TElm const* param)
{
	assert(param);

	return (size_t) (param[1]+0.5);
}

SO_TElm SO_LICEGamma(const SO_TElm *param)
{
	assert(param);

	return param[2];
}

/* ---------------------------------------------------------------- */

/* The overall structure of this function is:
 * - Retrieve variables from context and parameters.
 * - Allocate and initialize vectors and other data needed by this optimization method.
 * - Perform optimization.
 * - De-allocate data.
 * - Return best result from the optimization.
 */

SO_TFitness SO_LICE(const SO_TElm *param, void* context, const SO_TFitness fitnessLimit)
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

	/* Retrieve parameter specific to LICE method. */
	SO_TElm gamma = SO_LICEGamma(param);
	SO_TElm gamma2 = SO_LICEGamma2(param);
	const size_t N = SO_LICEN(param);

	/* Parameter for base-layer. */
	SO_TElm gammaTry = gamma;

	/* Allocate agent position and search-range vectors. */
	SO_TElm *x = SO_NewVector(n);					/* Current position. */
	SO_TElm *y = SO_NewVector(n);					/* Potentially new position. */
	SO_TElm *d = SO_NewVector(n);					/* Base-layer search-space. */

	/* Initialize search-range and decrease-factor. */
	SO_TElm r = 1;									/* Base-layer search-range. */

	/* Initialize meta-layer's search-range and decrease-factor. */
	SO_TElm d2 = SO_kLICEGammaUpper - SO_kLICEGammaLower;	/* Search-space. */
	SO_TElm r2 = 1;											/* Search-range. */
	SO_TElm q2 = pow(2.0, -1.0/gamma2);					/* Decrease-factor for one-dim problem. */

	/* Number of iterations for base-layer LUS. */
	size_t M = n*N;

	/* Iteration variables. */
	size_t i, j, k;

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

	/* Perform meta-layer iterations. */
	for (i=1; i<numIterations;)
	{
		/* Compute decrease factor for base-layer LUS. */
		SO_TElm q = pow(2.0, -1.0/(n*gammaTry));

		/* Bool indicating whether a fitness improvement has been found. */
		int improvementFound = 0;

		/* Perform base-layer optimization iterations. */
		for (j=0; i<numIterations && j<M; i++, j++)
		{
			/* Compute potentially new position. */
			for (k=0; k<n; k++)
			{
				/* Pick a sample from the neighbourhood of the current
				 * position and within the given range. The sample
				 * is assumed bounded by the SO_SampleBoundedOne() function. */
				y[k] = SO_SampleBoundedOne(x[k], r * d[k], lowerBound[k], upperBound[k]);
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

				/* Set a flag indicating fitness improvement was found. */
				improvementFound = 1;
			}
			else
			{
				/* Decrease the search-range. */
				r *= q;
			}

			/* Trace fitness of best found solution. */
			SO_SetFitnessTrace(c, i, fitness);
		}

		/* If fitness improvement found then keep gamma-parameter
		 * for base-layer LUS. */
		if (improvementFound)
		{
			gamma = gammaTry;
		}

		/* Decrease search-range of meta-layer LUS. */
		r2 *= q2;

		/* Sample new gamma parameter for base-layer LUS. */
		gammaTry = SO_SampleBoundedOne(gamma, r2 * d2, SO_kLICEGammaLower, SO_kLICEGammaUpper);
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
