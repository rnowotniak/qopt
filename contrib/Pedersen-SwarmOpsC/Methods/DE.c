/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	DE
 *
 *	See header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/DE.h>
#include <SwarmOps/Tools/Bound.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Matrix.h>
#include <SwarmOps/Tools/Init.h>
#include <SwarmOps/Tools/Random.h>
#include <RandomOps/RandomSet.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

/* The method has the following parameters:
 * Index 0: Number of agents NP
 * Index 1: Crossover Probability (CR)
 * Index 2: Differential weight (F)
 */

const char SO_kNameDE[] = "DE (Basic)";
const char* SO_kParameterNameDE[] = {"NP", "CR", "F"};

/* Parameters and boundaries for use in ANN Meta optimization. */
const SO_TElm SO_kParametersDefaultDE[SO_kNumParametersDE] = {172.0, 0.965609, 0.361520}; /* Meta-optimized parameters, 12 problems, 30 dim, 200*dim iterations. */
const SO_TElm SO_kParametersLowerDE[SO_kNumParametersDE] = {3, 0, 0};
const SO_TElm SO_kParametersUpperDE[SO_kNumParametersDE] = {200, 1, 2.0};


#if 0
/* Parameters and boundaries for use in Benchmark optimization. */
const SO_TElm SO_kParametersDefaultDE[SO_kNumParametersDE] = {50, 0.3, 0.75};	/* Recommended parameters. */
const SO_TElm SO_kParametersDefaultDE[SO_kNumParametersDE] = {50, 0.1, 0.5};	/* Hand-tuned parameters. */
const SO_TElm SO_kParametersDefaultDE[SO_kNumParametersDE] = {96, 0, 1.998473}; /* Meta-optimized parameters. */
const SO_TElm SO_kParametersDefaultDE[SO_kNumParametersDE] = {172.0, 0.965609, 0.361520}; /* Meta-optimized parameters, 12 problems, 30 dim, 200*dim iterations. */
#endif

/* ---------------------------------------------------------------- */

size_t SO_DENumAgents(SO_TElm const* param)
{
	assert(param);

	return (size_t) (param[0]+0.5);
}

SO_TElm SO_DECR(SO_TElm const* param)
{
	assert(param);

	return param[1];
}

SO_TElm SO_DEF(SO_TElm const* param)
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

SO_TFitness SO_DE(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit)
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

	/* Retrieve parameters specific to DE method. */
	size_t numAgents = SO_DENumAgents(param);
	SO_TElm CR = SO_DECR(param);
	SO_TElm F = SO_DEF(param);

	/* Allocate agent positions and associated fitnesses. */
	SO_TElm** agents = SO_NewMatrix(numAgents, n);
	SO_TFitness* agentFitness = SO_NewFitnessVector(numAgents);
	SO_TElm *t = SO_NewVector(n);

	/* Iteration variables. */
	size_t i, j, k;

	/* Fitness variables. */
	SO_TFitness gFitness, newFitness;
	SO_TElm* g = 0;

	/* Initialize all agents.
	 * This counts as iterations below. */
	for (j=0; j<numAgents && j<numIterations; j++)
	{
		/* Refer to the j'th agent as x. */
		SO_TElm* x = agents[j];

		/* Initialize agent-position in search-space. */
		SO_InitUniform(x, n, lowerInit, upperInit);

		/* Compute fitness of initial position. */
		agentFitness[j] = f(x, fContext, SO_kFitnessMax);

		/* Update swarm's best known position. */
		if (g == 0 || agentFitness[j]<gFitness)
		{
			g = x;
			gFitness = agentFitness[j];
		}

		/* Trace fitness of best found solution. */
		SO_SetFitnessTrace(c, j, gFitness);
	}

	for (i=numAgents; i<numIterations;)
	{
		assert(numAgents>0);

		for (j=0; j<numAgents && i<numIterations; j++, i++)
		{
			/* Refer to the j'th agent as x. */
			SO_TElm* x = agents[j];

			/* Pick random dimension. */
			SO_TDim R = SO_RandIndex(n);

			/* Other agents to be picked at random. */
			SO_TElm *a, *b;

			size_t R1, R2;

			/* Pick random and distinct agent-indices.
			 * Not necessarily distinct from x though. */
			SO_RandIndex2(numAgents, &R1, &R2);

			/* Refer to the randomly picked agents as a and b. */
			a = agents[R1];
			b = agents[R2];

			/* Store old position. */
			SO_CopyVector(t, x, n);

			/* Compute potentially new position. */
			for (k=0; k<n; k++)
			{
				if (SO_RandUni()<CR || k==R)
				{
					x[k] = g[k] + F * (a[k] - b[k]);
				}
			}

			/* Enforce bounds before computing new fitness. */
			SO_BoundAll(x, n, lowerBound, upperBound);

			/* Compute new fitness. */
			newFitness = f(x, fContext, agentFitness[j]);

			/* Update agent in case of fitness improvement. */
			if (newFitness < agentFitness[j])
			{
				/* Update agent's fitness. Position is already updated. */
				agentFitness[j] = newFitness;

				/* Update swarm's best known position. */
				if (newFitness < gFitness)
				{
					g = x;
					gFitness = newFitness;
				}
			}
			else /* Fitness was not an improvement. */
			{
				/* Restore old position. */
				SO_CopyVector(x, t, n);
			}

			/* Trace fitness of best found solution. */
			SO_SetFitnessTrace(c, i, gFitness);
		}
	}

	/* Set best position found in this run. */
	SO_MethodSetResult(c, g, gFitness);

	/* Update all-time best known position. */
	SO_MethodUpdateBest(c, g, gFitness);

	/* Delete agent-position-matrix and fitness-vector. */
	SO_FreeMatrix(agents, numAgents);
	SO_FreeVector(agentFitness);
	SO_FreeVector(t);

	return gFitness;
}

/* ---------------------------------------------------------------- */
