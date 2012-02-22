/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	MYG
 *
 *	See header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/MYG.h>
#include <SwarmOps/Tools/Bound.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Matrix.h>
#include <SwarmOps/Tools/Init.h>
#include <SwarmOps/Tools/Random.h>
#include <RandomOps/RandomSet.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

const char SO_kNameMYG[] = "MYG";
const char* SO_kParameterNameMYG[] = {"NP", "F"};

/* The method has the following parameters:
 * Index 0: Number of agents NP
 * Index 1: Differential weight F, aka. alpha.
 */

/* Parameters and boundaries for use in ANN Meta optimization. */
const SO_TElm SO_kParametersDefaultMYG[SO_kNumParametersMYG] = {300, 1.627797};	/* Meta-optimized to ANN. */
const SO_TElm SO_kParametersLowerMYG[SO_kNumParametersMYG] = {5, 0.5};
const SO_TElm SO_kParametersUpperMYG[SO_kNumParametersMYG] = {300, 2};

/* ---------------------------------------------------------------- */

size_t SO_MYGNumAgents(SO_TElm const* param)
{
	assert(param);

	return (size_t) (param[0]+0.5);
}

SO_TElm SO_MYGF(SO_TElm const* param)
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

SO_TFitness SO_MYG(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit)
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

	/* Retrieve parameters specific to MYG method. */
	size_t numAgents = SO_MYGNumAgents(param);
	SO_TElm F = SO_MYGF(param);

	/* Allocate agent positions and associated fitnesses. */
	SO_TElm** agents = SO_NewMatrix(numAgents, n);
	SO_TFitness* agentFitness = SO_NewFitnessVector(numAgents);
	SO_TElm *y = SO_NewVector(n);

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
		/* Refer to the j'th agent as x. */
		SO_TElm* x = agents[j];

		/* Initialize agent-position in search-space. */
		SO_InitUniform(x, n, lowerInit, upperInit);

		/* Compute fitness of initial position. */
		agentFitness[j] = f(x, fContext, SO_kFitnessMax);

		/* Update swarm's best known position. */
		if (agentFitness[j]<*gFitness)
		{
			SO_CopyVector(g, x, n);
			*gFitness = agentFitness[j];
		}

		/* Trace fitness of best found solution. */
		SO_SetFitnessTrace(c, j, *gFitness);
	}

	/* Perform actual optimization iterations. */
	for (i=numAgents; i<numIterations; i++)
	{
		j = SO_RandIndex(numAgents);

		{
			/* Refer to the j'th agent as x. */
			SO_TElm* x = agents[j];

			/* Other agents to be picked at random. */
			SO_TElm *a, *b;

			size_t R1, R2;

			/* Pick random and distinct agent-indices.
			 * Not necessarily distinct from x though. */
			SO_RandIndex2(numAgents, &R1, &R2);

			/* Refer to the randomly picked agents as a and b. */
			a = agents[R1];
			b = agents[R2];

			/* Compute potentially new position. */
			for (k=0; k<n; k++)
			{
				y[k] = g[k] + F * (a[k] - b[k]);
			}

			/* Enforce bounds before computing new fitness. */
			SO_BoundAll(y, n, lowerBound, upperBound);

			/* Compute new fitness. */
			newFitness = f(y, fContext, agentFitness[j]);

			/* Update agent in case of fitness improvement. */
			if (newFitness < agentFitness[j])
			{
				/* Update agent's position. */
				SO_CopyVector(x, y, n);

				/* Update agent's fitness. */
				agentFitness[j] = newFitness;

				/* Update swarm's best known position. */
				if (newFitness < *gFitness)
				{
					/* Update best-known position. */
					SO_CopyVector(g, x, n);
					*gFitness = newFitness;
				}
			}

			/* Trace fitness of best found solution. */
			SO_SetFitnessTrace(c, i, *gFitness);
		}
	}

	/* Update all-time best known position. */
	SO_MethodUpdateBest(c, g, *gFitness);

	/* Delete agent-position-matrix and fitness-vector. */
	SO_FreeMatrix(agents, numAgents);
	SO_FreeVector(agentFitness);
	SO_FreeVector(y);

	return *gFitness;
}

/* ---------------------------------------------------------------- */
