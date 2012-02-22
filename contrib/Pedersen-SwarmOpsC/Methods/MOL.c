/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	MOL
 *
 *	See header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/MOL.h>
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
 * Index 2: Weight on swarm-best attraction
 */

const char SO_kNameMOL[] = "MOL";
const char* SO_kParameterNameMOL[] = {"S", "omega", "phi_g"};

/* Parameters and boundaries for use with the 'Simplifying PSO' paper. */
const SO_TElm SO_kParametersDefaultMOL[SO_kNumParametersMOL] = {153.0, -0.289623, 1.494742}; /* Meta-optimized for the ANN Cancer and Card problems. */
const SO_TElm SO_kParametersLowerMOL[SO_kNumParametersMOL] = {1.0, -2.0, -4.0};
const SO_TElm SO_kParametersUpperMOL[SO_kNumParametersMOL] = {200.0, 2.0, 4.0};

#if 0
const SO_TElm SO_kParametersDefaultMOL[SO_kNumParametersMOL] = {74.0, -0.265360, 1.612467}; /* Meta-optimized for ANN Cancer problem. */
#endif

/* ---------------------------------------------------------------- */

size_t SO_MOLNumAgents(SO_TElm const* param)
{
	assert(param);

	return (size_t) (param[0]+0.5);
}

SO_TElm SO_MOLOmega(SO_TElm const* param)
{
	assert(param);

	return param[1];
}

SO_TElm SO_MOLPhi(SO_TElm const* param)
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

SO_TFitness SO_MOL(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit)
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

	/* Retrieve parameters specific to MOL method. */
	size_t numAgents = SO_MOLNumAgents(param);
	SO_TElm omega = SO_MOLOmega(param);
	SO_TElm phi = SO_MOLPhi(param);

	/* Allocate agent positions and associated fitnesses. */
	SO_TElm **agents = SO_NewMatrix(numAgents, n);
	SO_TElm **velocities = SO_NewMatrix(numAgents, n);

	/* Allocate velocity boundaries. */
	SO_TElm *velocityLowerBound = SO_NewVector(n);
	SO_TElm *velocityUpperBound = SO_NewVector(n);

	/* Iteration variables. */
	size_t i, j, k;

	/* Fitness variable. */
	SO_TFitness newFitness;

	/* Initialize best-known fitness to its worst possible value. */
	*gFitness = SO_kFitnessMax;

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

		/* Compute fitness of initial position.
		 * Note the use of pre-emptive fitness evaluation,
		 * by having the pre-emptive limit gFitness. */
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
			/* Refer to the j'th agent as x and v. */
			SO_TElm *x = agents[j];
			SO_TElm *v = velocities[j];

			/* Pick random weights. */
			SO_TElm r = SO_RandUni();

			/* Update velocity. */
			for (k=0; k<n; k++)
			{
				v[k] = omega * v[k] + phi * r * (g[k] - x[k]);
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

	/* Delete agent-position-matrix and fitness-vector. */
	SO_FreeMatrix(agents, numAgents);
	SO_FreeMatrix(velocities, numAgents);

	/* Delete velocity boundaries. */
	SO_FreeVector(velocityLowerBound);
	SO_FreeVector(velocityUpperBound);

	return *gFitness;
}

/* ---------------------------------------------------------------- */
