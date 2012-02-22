/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	DETP
 *
 *	See header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/DETP.h>
#include <SwarmOps/Tools/Bound.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Matrix.h>
#include <SwarmOps/Tools/Init.h>
#include <SwarmOps/Tools/Random.h>
#include <RandomOps/RandomSet.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

/* Determine crossover-name. */
#if (SO_kDETPCrossover == SO_kDECrossoverRand1Bin)
#define SO_kNameDETPCrossover SO_kNameDECrossoverRand1Bin
#elif (SO_kDETPCrossover == SO_kDECrossoverRand1BinEO)
#define SO_kNameDETPCrossover SO_kNameDECrossoverRand1BinEO
#elif (SO_kDETPCrossover == SO_kDECrossoverBest1Bin)
#define SO_kNameDETPCrossover SO_kNameDECrossoverBest1Bin
#elif (SO_kDETPCrossover == SO_kDECrossoverBest1BinSimple)
#define SO_kNameDETPCrossover SO_kNameDECrossoverBest1BinSimple
#endif

/* Determine period-name. */
#if (SO_kDETPNumPeriods == 1)
#define SO_kNameDETPPeriod "1"
const char* SO_kParameterNameDETP[] = {"NP", "CR_1", "F_1"};
#elif (SO_kDETPNumPeriods == 2)
#define SO_kNameDETPPeriod "2"
const char* SO_kParameterNameDETP[] = {"NP", "CR_1", "CR_2", "F_1", "F_2"};
#elif (SO_kDETPNumPeriods == 4)
#define SO_kNameDETPPeriod "4"
const char* SO_kParameterNameDETP[] = {"NP", "CR_1", "CR_2", "CR_3", "CR_4", "F_1", "F_2", "F_3", "F_4"};
#elif (SO_kDETPNumPeriods == 8)
#define SO_kNameDETPPeriod "8"
const char* SO_kParameterNameDETP[] = {"NP", "CR_1", "CR_2", "CR_3", "CR_4", "CR_5", "CR_6", "CR_7", "CR_8", "F_1", "F_2", "F_3", "F_4", "F_5", "F_6", "F_7", "F_8"};
#else
#define SO_kNameDETPPeriod ""
#endif

const char SO_kNameDETP[] = "DETP" SO_kNameDETPPeriod SO_kNameDETPCrossover;

/* ---------------------------------------------------------------- */

/* The method has the following parameters:
 * Index 0: Number of agents
 * Index 1: Crossover probability (CR1)
 * Index 2: Crossover probability (CR2)
 * Index 3: ...
 * Index k+0: Differential weight (F1)
 * Index k+1: Differential weight (F2)
 * Index k+2: ...
 */

/* Parameter defaults and boundaries. */
#if SO_kDETPNumPeriods==1
const SO_TElm SO_kParametersDefaultDETP[SO_kNumParametersDETP] = {44, 0.967665, 0.536893};
const SO_TElm SO_kParametersLowerDETP[SO_kNumParametersDETP] = {4, 0, 0};
const SO_TElm SO_kParametersUpperDETP[SO_kNumParametersDETP] = {200, 1, 2.0};
#elif SO_kDETPNumPeriods==2
const SO_TElm SO_kParametersDefaultDETP[SO_kNumParametersDETP] = {9.0, 0.040135, 0.576005, 0.955493, 0.320264};
const SO_TElm SO_kParametersLowerDETP[SO_kNumParametersDETP] = {4, 0, 0, 0, 0};
const SO_TElm SO_kParametersUpperDETP[SO_kNumParametersDETP] = {200, 1, 1, 2.0, 2.0};
#elif SO_kDETPNumPeriods==4
const SO_TElm SO_kParametersDefaultDETP[SO_kNumParametersDETP] = {44, 0.967665, 0.967665, 0.967665, 0.967665, 0.536893, 0.536893, 0.536893, 0.536893};
const SO_TElm SO_kParametersLowerDETP[SO_kNumParametersDETP] = {4, 0, 0, 0, 0, 0, 0, 0, 0};
const SO_TElm SO_kParametersUpperDETP[SO_kNumParametersDETP] = {200, 1, 1, 1, 1, 2.0, 2.0, 2.0, 2.0};
#elif SO_kDETPNumPeriods==8
const SO_TElm SO_kParametersDefaultDETP[SO_kNumParametersDETP] = {44, 0.967665, 0.967665, 0.967665, 0.967665, 0.967665, 0.967665, 0.967665, 0.967665, 0.536893, 0.536893, 0.536893, 0.536893, 0.536893, 0.536893, 0.536893, 0.536893};
const SO_TElm SO_kParametersLowerDETP[SO_kNumParametersDETP] = {4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
const SO_TElm SO_kParametersUpperDETP[SO_kNumParametersDETP] = {200, 1, 1, 1, 1, 1, 1, 1, 1, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
#endif

/* ---------------------------------------------------------------- */

size_t SO_DETPNumAgents(SO_TElm const* param)
{
	assert(param);

	return (size_t) (param[0]+0.5);
}

/* Return the parameter index offset for the F and CR pair of parameters
 * corresponding to the i'th iteration out of a total kNumIterations. */
size_t SO_DETPPeriod(const size_t i, const size_t kNumIterations)
{
	double periodD;
	size_t periodI;

	periodD = (double) i / kNumIterations;
	periodD = SO_Bound(periodD, 0, 1);
	periodI = (size_t) (periodD*SO_kDETPNumPeriods);

	assert(periodI>=0 && periodI<SO_kDETPNumPeriods);

	return periodI;
}

SO_TElm SO_DETPCrossoverP(SO_TElm const* param, const size_t i, const size_t kNumIterations)
{
	size_t period, index;

	assert(param);

	period = SO_DETPPeriod(i, kNumIterations);
	index = 1+period;

	assert(index>=0 && index<SO_kNumParametersDETP);

	return param[index];
}

SO_TElm SO_DETPF(SO_TElm const* param, const size_t i, const size_t kNumIterations)
{
	size_t period, index;

	assert(param);

	period = SO_DETPPeriod(i, kNumIterations);
	index = 1+SO_kDETPNumPeriods+period;

	assert(index>=0 && index<SO_kNumParametersDETP);

	return param[index];
}

/* ---------------------------------------------------------------- */

/* The overall structure of this function is:
 * - Retrieve variables from context and parameters.
 * - Allocate and initialize vectors and other data needed by this optimization method.
 * - Perform optimization.
 * - De-allocate data.
 * - Return best result from the optimization.
 */

SO_TFitness SO_DETP(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit)
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
	size_t numAgents = SO_DETPNumAgents(param);

	/* Allocate agent positions and associated fitnesses. */
	SO_TElm** agents = SO_NewMatrix(numAgents, n);
	SO_TFitness* agentFitness = SO_NewFitnessVector(numAgents);
	SO_TElm *t = SO_NewVector(n);
	SO_TElm *w = SO_NewVector(n);

	/* Iteration variables. */
	size_t i, j;

	/* Fitness variables. */
	SO_TFitness gFitness, newFitness;
	SO_TElm* g = 0;

	/* Random set for picking distinct agents. */
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

	for (i=numAgents; i<numIterations;)
	{
		SO_TElm CR = SO_DETPCrossoverP(param, i, numIterations);
		SO_TElm F = SO_DETPF(param, i, numIterations);

		/* Initialize crossover-weight vector. Vector-based. */
		SO_InitVector(w, F, n);

		assert(numAgents>0);

		for (j=0; j<numAgents && i<numIterations; j++, i++)
		{
			/* Refer to the j'th agent as x. */
			SO_TElm* x = agents[j];

			/* Store old position. */
			SO_CopyVector(t, x, n);

			/* Reset the random-set used for picking distinct agents.
			 * Exclude the j'th agent (also referred to as x). */
			RO_RandSetResetExclude(&randSet, j);

			/* Perform DE crossover. */
			SO_DEEngineCrossover(SO_kDETPCrossover, n, CR, w, x, g, numAgents, agents, &randSet);

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
	SO_FreeVector(w);

	/* Free random set */
	RO_RandSetFree(&randSet);

	return gFitness;
}

/* ---------------------------------------------------------------- */
