/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	FAE
 *
 *	See header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/FAE.h>
#include <SwarmOps/Tools/Bound.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Matrix.h>
#include <SwarmOps/Tools/Init.h>
#include <SwarmOps/Tools/Random.h>
#include <SwarmOps/Tools/Memory.h>
#include <SwarmOps/Tools/Denormal.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

/* The method has the following parameters:
 * Index 0: Number of agents
 * Index 1: Lambda-g
 * Index 2: Lambda-x
 */

const char SO_kNameFAE[] = "FAE";
const char* SO_kParameterNameFAE[] = {"S", "lambda_g", "lambda_x"};

/* Parameters and boundaries for use in ANN Meta optimization. */
const SO_TElm SO_kParametersDefaultFAE[SO_kNumParametersFAE] = {100.000000, 1.486496, -3.949617};
const SO_TElm SO_kParametersLowerFAE[SO_kNumParametersFAE] = {1.0, -2.0, -8.0};
const SO_TElm SO_kParametersUpperFAE[SO_kNumParametersFAE] = {100.0, 2.0, -1.0};

#if 0
/* Parameters and boundaries for use in Benchmark optimization. */
const SO_TElm SO_kParametersDefaultFAE[SO_kNumParametersFAE] = {100.0, 0.669848, -2.870105};
const SO_TElm SO_kParametersLowerFAE[SO_kNumParametersFAE] = {1.0, -2.0, -8.0};
const SO_TElm SO_kParametersUpperFAE[SO_kNumParametersFAE] = {100.0, 2.0, 6.0};
#endif

/* ---------------------------------------------------------------- */

size_t SO_FAENumAgents(SO_TElm const* param)
{
	assert(param);

	return (size_t) (param[0]+0.5);
}

SO_TElm SO_FAELambdaG(SO_TElm const* param)
{
	assert(param);

	return param[1];
}

SO_TElm SO_FAELambdaX(SO_TElm const* param)
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

SO_TFitness SO_FAE(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit)
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

	/* Retrieve parameters specific to FAE method. */
	size_t numAgents = SO_FAENumAgents(param);
	SO_TElm lambdaG = SO_FAELambdaG(param);
	SO_TElm lambdaX = SO_FAELambdaX(param);

	/* Allocate matrix for agent positions. */
	SO_TElm **agents = SO_NewMatrix(numAgents, n);

	/* Iteration variables. */
	size_t i, j, k;

	/* Fitness variable. */
	SO_TFitness newFitness;

	/* Initialize best-known fitness to its worst possible value. */
	*gFitness = SO_kFitnessMax;

	/* Initialize all agents.
	 * This counts as iterations below. */
	for (j=0; j<numAgents && j<numIterations; j++)
	{
		/* Refer to the j'th agent as x and v. */
		SO_TElm *x = agents[j];

		/* Initialize agent-position in search-space. */
		SO_InitUniform(x, n, lowerInit, upperInit);

		/* Compute fitness of initial position.
		 * Note the use of pre-emptive fitness evaluation,
		 * by having the limit gFitness. This works
		 * because FAE is non-greedy. */
		newFitness = f(x, fContext, *gFitness);

		/* Update swarm's best known position. */
		if (newFitness<*gFitness)
		{
			SO_CopyVector(g, x, n);
			*gFitness = newFitness;
		}

		/* Trace fitness of best found solution. */
		SO_SetFitnessTrace(c, j, *gFitness);
	}

	/* Perform actual optimization iterations. */
	for (i=numAgents; i<numIterations; i++)
	{
		/* Pick random agent. */
		j = SO_RandIndex(numAgents);

		{
			/* Refer to the j'th agent as x. */
			SO_TElm *x = agents[j];

			/* Pick random weights. */
			SO_TElm rG = SO_RandUni();
			SO_TElm rX = SO_RandUni();

			/* Update position. */
			for (k=0; k<n; k++)
			{
				x[k] = rG * lambdaG * g[k] + rX * lambdaX * x[k];
			}

			/* Enforce bounds before computing new fitness. */
			SO_BoundAll(x, n, lowerBound, upperBound);

			/* Compute new fitness. */
			newFitness = f(x, fContext, *gFitness);

			/* Update best-known position in case of fitness improvement. */
			if (newFitness < *gFitness)
			{
				/* Update best-known position. */
				SO_CopyVector(g, x, n);
				*gFitness = newFitness;
			}

			/* Trace fitness of best found solution. */
			SO_SetFitnessTrace(c, i, *gFitness);
		}
	}

	/* Update all-time best known position. */
	SO_MethodUpdateBest(c, g, *gFitness);

	/* Delete agent-position-matrix. */
	SO_FreeMatrix(agents, numAgents);

	return *gFitness;
}

/* ---------------------------------------------------------------- */
