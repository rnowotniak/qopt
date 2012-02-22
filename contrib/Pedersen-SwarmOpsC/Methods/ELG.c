/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	ELG
 *
 *	See header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/ELG.h>
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
 */

const char SO_kNameELG[] = "ELG";
const char* SO_kParameterNameELG[] = {"NP"};

const SO_TElm SO_kParametersDefaultELG[SO_kNumParametersELG] = {143.0};
const SO_TElm SO_kParametersLowerELG[SO_kNumParametersELG] = {2};
const SO_TElm SO_kParametersUpperELG[SO_kNumParametersELG] = {170};

/* ---------------------------------------------------------------- */

size_t SO_ELGNumAgents(SO_TElm const* param)
{
	assert(param);

	return (size_t) (param[0]+0.5);
}

/* ---------------------------------------------------------------- */

/* The overall structure of this function is:
 * - Retrieve variables from context and parameters.
 * - Allocate and initialize vectors and other data needed by this optimization method.
 * - Perform optimization.
 * - De-allocate data.
 * - Return best result from the optimization.
 */

SO_TFitness SO_ELG(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit)
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

	/* Retrieve parameters specific to ELG method. */
	size_t numAgents = SO_ELGNumAgents(param);

	/* Allocate agent positions and associated fitnesses. */
	SO_TElm** agents = SO_NewMatrix(numAgents, n);
	SO_TFitness* agentFitness = SO_NewFitnessVector(numAgents);

	/* Iteration variables. */
	size_t i, j;

	/* Fitness variables. */
	SO_TFitness gFitness, newFitness;
	SO_TElm* g = 0;

	/* Allocate a random-set used for picking different agents. */
	struct RO_RandSet randSet = RO_RandSetInit(numAgents);

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

	/* Perform actual optimization iterations. */
	for (i=numAgents; i<numIterations; i++)
	{
		j = SO_RandIndex(numAgents);

		{
			/* Refer to the current agent as x. */
			SO_TElm* x = agents[j];

			/* Pick two random agents, not necessarily distinct. */
			size_t R1 = SO_RandIndex(numAgents);
			size_t R2 = SO_RandIndex(numAgents);

			/* Refer to the randomly picked agents as a and b. */
			SO_TElm *a = agents[R1];
			SO_TElm *b = agents[R2];

			/* Pick random dimension. */
			SO_TDim R = SO_RandIndex(n);

			/* Store old value for that dimension. */
			SO_TElm t = x[R];

			/* Update position for that dimension. */
			x[R] = g[R] + (a[R] - b[R]);

			/* Enforce bounds before computing new fitness. */
			SO_BoundOne(x, R, lowerBound, upperBound);

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
				x[R] = t;
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

	/* Delete random-set. */
	RO_RandSetFree(&randSet);

	return gFitness;
}

/* ---------------------------------------------------------------- */
