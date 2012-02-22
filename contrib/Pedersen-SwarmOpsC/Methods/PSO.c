/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	PSO
 *
 *	See header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/PSO.h>
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
 * Index 1: Inertia weight
 * Index 2: Weight on particle-best attraction
 * Index 3: Weight on swarm-best attraction
 */

const char SO_kNamePSO[] = "PSO (Basic)";
const char* SO_kParameterNamePSO[] = {"S", "omega", "phi_p", "phi_g"};

const SO_TElm SO_kParametersDefaultPSO[SO_kNumParametersPSO] = {148.0, -0.046644, 2.882152, 1.857463};	/* Tuned for ANN Cancer and Card. */
const SO_TElm SO_kParametersLowerPSO[SO_kNumParametersPSO] = {1.0, -2.0, -4.0, -4.0};
const SO_TElm SO_kParametersUpperPSO[SO_kNumParametersPSO] = {200.0, 2.0, 4.0, 4.0};

#if 0
/* Hand-tuned parameters */
const SO_TElm SO_kParametersDefaultPSO[SO_kNumParametersPSO] = {50.0, 0.729, 1.49445, 1.49445};
#endif

/* ---------------------------------------------------------------- */

size_t SO_PSONumAgents(SO_TElm const* param)
{
	assert(param);

	return (size_t) (param[0]+0.5);
}

SO_TElm SO_PSOOmega(SO_TElm const* param)
{
	assert(param);

	return param[1];
}

SO_TElm SO_PSOPhi1(SO_TElm const* param)
{
	assert(param);

	return param[2];
}

SO_TElm SO_PSOPhi2(SO_TElm const* param)
{
	assert(param);

	return param[3];
}

/* ---------------------------------------------------------------- */

/* The overall structure of this function is:
 * - Retrieve variables from context and parameters.
 * - Allocate and initialize vectors and other data needed by this optimization method.
 * - Perform optimization.
 * - De-allocate data.
 * - Return best result from the optimization.
 */

SO_TFitness SO_PSO(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit)
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

	/* Retrieve parameters specific to PSO method. */
	size_t numAgents = SO_PSONumAgents(param);
	SO_TElm omega = SO_PSOOmega(param);
	SO_TElm phi1 = SO_PSOPhi1(param);
	SO_TElm phi2 = SO_PSOPhi2(param);

	/* Allocate agent positions and associated fitnesses. */
	SO_TElm **agents = SO_NewMatrix(numAgents, n);
	SO_TElm **velocities = SO_NewMatrix(numAgents, n);
	SO_TElm **bestAgentPosition = SO_NewMatrix(numAgents, n);
	SO_TFitness *bestAgentFitness = SO_NewFitnessVector(numAgents);

	/* Allocate velocity boundaries. */
	SO_TElm *velocityLowerBound = SO_NewVector(n);
	SO_TElm *velocityUpperBound = SO_NewVector(n);

	/* Iteration variables. */
	size_t i, j, k;

	/* Fitness variables. */
	SO_TFitness gFitness, newFitness;
	SO_TElm* g = 0;

	/* Initialize velocity boundaries. */
	for (k=0; k<n; k++)
	{
		SO_TElm range = fabs(upperBound[k]-lowerBound[k]);

		velocityLowerBound[k] = -range;
		velocityUpperBound[k] = range;
	}

	/* Initialize all agents.
	 * This counts as iterations below. */
	for (j=0; j<numAgents && j<numIterations; j++)
	{
		/* Refer to the j'th agent as x and v. */
		SO_TElm *x = agents[j];
		SO_TElm *v = velocities[j];

		/* Initialize agent-position in search-space. */
		SO_InitUniform(x, n, lowerInit, upperInit);

		/* Initialize velocity. */
		SO_InitUniform(v, n, velocityLowerBound, velocityUpperBound);

		/* Compute fitness of initial position. */
		bestAgentFitness[j] = f(x, fContext, SO_kFitnessMax);

		/* Initialize best known position.
		 * Contents must be copied because the agent
		 * will likely move to worse positions. */
		SO_CopyVector(bestAgentPosition[j], x, n);

		/* Update swarm's best known position.
		 * This must reference the agent's best-known
		 * position because the current position changes. */
		if (g == 0 || bestAgentFitness[j]<gFitness)
		{
			g = bestAgentPosition[j];
			gFitness = bestAgentFitness[j];
		}

		/* Trace fitness of best found solution. */
		SO_SetFitnessTrace(c, j, gFitness);
	}

	/* Perform actual optimization iterations. */
	for (i=numAgents; i<numIterations;)
	{
		assert(numAgents>0);

		for (j=0; j<numAgents && i<numIterations; j++, i++)
		{
			/* Refer to the j'th agent as x and v. */
			SO_TElm *x = agents[j];
			SO_TElm *v = velocities[j];
			SO_TElm *p = bestAgentPosition[j];

			/* Pick random weights. */
			SO_TElm r1 = SO_RandUni();
			SO_TElm r2 = SO_RandUni();

			/* Update velocity. */
			for (k=0; k<n; k++)
			{
				v[k] = omega * v[k] + phi1 * r1 * (p[k] - x[k]) + phi2 * r2 * (g[k] - x[k]);
			}

			/* Fix denormalized floating-point values in velocity. */
			SO_DenormalFixAll(v, n);

			/* Enforce velocity bounds before updating position. */
			SO_BoundAll(v, n, velocityLowerBound, velocityUpperBound);

			/* Update position. */
			for (k=0; k<n; k++)
			{
				x[k] = x[k] + v[k];
			}

			/* Enforce bounds before computing new fitness. */
			SO_BoundAll(x, n, lowerBound, upperBound);

			/* Compute new fitness. */
			newFitness = f(x, fContext, bestAgentFitness[j]);

			/* Update best-known position in case of fitness improvement. */
			if (newFitness < bestAgentFitness[j])
			{
				/* Update best-known position.
				 * Contents must be copied because the agent
				 * will likely move to worse positions. */
				SO_CopyVector(bestAgentPosition[j], x, n);
				bestAgentFitness[j] = newFitness;

				/* Update swarm's best known position.
				 * This must reference the agent's best-known
				 * position because the current position changes. */
				if (newFitness < gFitness)
				{
					g = bestAgentPosition[j];
					gFitness = bestAgentFitness[j];
				}
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
	SO_FreeMatrix(velocities, numAgents);
	SO_FreeMatrix(bestAgentPosition, numAgents);
	SO_FreeVector(bestAgentFitness);

	/* Delete velocity boundaries. */
	SO_FreeVector(velocityLowerBound);
	SO_FreeVector(velocityUpperBound);

	return gFitness;
}

/* ---------------------------------------------------------------- */
