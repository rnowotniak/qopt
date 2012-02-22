/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	JDE
 *
 *	See header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/JDE.h>
#include <SwarmOps/Tools/Bound.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Matrix.h>
#include <SwarmOps/Tools/Init.h>
#include <SwarmOps/Tools/Random.h>
#include <RandomOps/RandomSet.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

/* Determine crossover-name. */
#if (SO_kJDECrossover == SO_kDECrossoverRand1Bin)
#define SO_kNameJDECrossover SO_kNameDECrossoverRand1Bin
#elif (SO_kJDECrossover == SO_kDECrossoverRand1BinEO)
#define SO_kNameJDECrossover SO_kNameDECrossoverRand1BinEO
#elif (SO_kJDECrossover == SO_kDECrossoverBest1Bin)
#define SO_kNameJDECrossover SO_kNameDECrossoverBest1Bin
#elif (SO_kJDECrossover == SO_kDECrossoverBest1BinSimple)
#define SO_kNameJDECrossover SO_kNameDECrossoverBest1BinSimple
#endif

const char SO_kNameJDE[] = "JDE" SO_kNameJDECrossover;
const char* SO_kParameterNameJDE[] = {"NP", "F_{init}", "F_l", "F_u", "tau_{F}", "CR_{init}", "CR_l", "CR_u", "tau_{CR}"};

/* ---------------------------------------------------------------- */

/* The method has the following parameters:
 * Index 0: Number of agents
 * Index 1: FInit (Differential weight, initial value)
 * Index 2: Fl
 * Index 3: Fu
 * Index 4: TauF (aka. Tau1)
 * Index 5: CRInit (Crossover probability, initial value)
 * Index 6: CRl
 * Index 7: CRu
 * Index 8: TauCR (aka. Tau2)
 */

/* Parameters and boundaries. */

#if (SO_kJDECrossover == SO_kDECrossoverRand1Bin)

#if 0
/* Tuned for benchmarks when allowed dim*5000 iterations. */
const SO_TElm SO_kParametersDefaultJDE[SO_kNumParametersJDE] = {59.0, 0.568105, 0.390495, 0.592659, 0.293519, 0.016715, 0.377906, 0.284746, 0.069810};
#elif 0
/* Tuned for 12 benchmarks when allowed dim*200 iterations. */
const SO_TElm SO_kParametersDefaultJDE[SO_kNumParametersJDE] = {16.0, 0.500358, 0.419994, 0.621257, 0.573597, 0.573335, 0.128144, 0.871238, 0.705309};
#elif 0
/* Tuned for 3 benchmarks when allowed dim*200 iterations. */
const SO_TElm SO_kParametersDefaultJDE[SO_kNumParametersJDE] = {8.0, 0.453133, 0.247631, 1.548331, 0.659707, 0.847650, 0.104456, 0.122205, 0.875351};
#elif 0
/* Tuned for Schwefel1-2 when allowed dim*200 iterations. */
const SO_TElm SO_kParametersDefaultJDE[SO_kNumParametersJDE] = {31.0, 1.728003, 0.321481, 1.379208, 0.093426, 0.970131, 0.976761, 0.023239, 0.501746};
#elif 1
/* Tuned for ANN Cancer problem when allowed dim*20 iterations. */
const SO_TElm SO_kParametersDefaultJDE[SO_kNumParametersJDE] = {9.0, 0.946048, 0.400038, 0.807296, 0.242418, 0.658907, 0.835000, 0.034128, 0.774923};
#else
/* Default parameters from jDE paper and source-code. */
const SO_TElm SO_kParametersDefaultJDE[SO_kNumParametersJDE] = {100, 0.5, 0.1, 0.9, 0.1, 0.9, 0, 1, 0.1};
#endif

#elif (SO_kJDECrossover == SO_kDECrossoverBest1BinSimple)

/* Tuned for ANN Cancer problem when allowed dim*20 iterations. */
const SO_TElm SO_kParametersDefaultJDE[SO_kNumParametersJDE] = {18.0, 1.393273, 0.319121, 0.933712, 0.619482, 0.777215, 0.889368, 0.160088, 0.846782};

#else
/* Default parameters from jDE paper and source-code. */
const SO_TElm SO_kParametersDefaultJDE[SO_kNumParametersJDE] = {100, 0.5, 0.1, 0.9, 0.1, 0.9, 0, 1, 0.1};
#endif

const SO_TElm SO_kParametersLowerJDE[SO_kNumParametersJDE] = {4, 0, 0, 0, 0, 0, 0, 0, 0};
const SO_TElm SO_kParametersUpperJDE[SO_kNumParametersJDE] = {200, 2.0, 2.0, 2.0, 1, 1, 1, 1, 1};

/* ---------------------------------------------------------------- */

size_t SO_JDENumAgents(SO_TElm const* param)
{
	assert(param);

	return (size_t) (param[0]+0.5);
}

SO_TElm SO_JDEFInit(SO_TElm const* param)
{
	assert(param);

	return param[1];
}

SO_TElm SO_JDEFl(SO_TElm const* param)
{
	assert(param);

	return param[2];
}

SO_TElm SO_JDEFu(SO_TElm const* param)
{
	assert(param);

	return param[3];
}

SO_TElm SO_JDETauF(SO_TElm const* param)
{
	assert(param);

	return param[4];
}

SO_TElm SO_JDECRInit(SO_TElm const* param)
{
	assert(param);

	return param[5];
}

SO_TElm SO_JDECRl(SO_TElm const* param)
{
	assert(param);

	return param[6];
}

SO_TElm SO_JDECRu(SO_TElm const* param)
{
	assert(param);

	return param[7];
}

SO_TElm SO_JDETauCR(SO_TElm const* param)
{
	assert(param);

	return param[8];
}

/* ---------------------------------------------------------------- */

/* The overall structure of this function is:
 * - Retrieve variables from context and parameters.
 * - Allocate and initialize vectors and other data needed by this optimization method.
 * - Perform optimization.
 * - De-allocate data.
 * - Return best result from the optimization.
 */

SO_TFitness SO_JDE(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit)
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
	size_t numAgents = SO_JDENumAgents(param);

	SO_TElm FInit = SO_JDEFInit(param);
	SO_TElm Fl = SO_JDEFl(param);
	SO_TElm Fu = SO_JDEFu(param);
	SO_TElm tauF = SO_JDETauF(param);

	SO_TElm CRInit = SO_JDECRInit(param);
	SO_TElm CRl = SO_JDECRl(param);
	SO_TElm CRu = SO_JDECRu(param);
	SO_TElm tauCR = SO_JDETauCR(param);

	/* Allocate agent positions and associated fitnesses. */
	SO_TElm** agents = SO_NewMatrix(numAgents, n);
	SO_TFitness* agentFitness = SO_NewFitnessVector(numAgents);
	SO_TElm *t = SO_NewVector(n);
	SO_TElm *w = SO_NewVector(n);
	SO_TElm *F = SO_NewVector(numAgents);
	SO_TElm *CR = SO_NewVector(numAgents);

	/* Iteration variables. */
	size_t i, j;

	/* Fitness variables. */
	SO_TFitness gFitness, newFitness;
	SO_TElm* g = 0;

	/* Random set for picking distinct agents. */
	struct RO_RandSet randSet = RO_RandSetInit(numAgents);

	/* Initialize 'self-adaptive' parameters. */
	SO_InitVector(F, FInit, numAgents);
	SO_InitVector(CR, CRInit, numAgents);

	/* Adjust CR parameters to remain within [0,1] */
	if (CRl+CRu>1)
	{
		CRu = 1 - CRl;
	}

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

			/* JDE 'Self-adaptive' parameters. */
			SO_TElm newF = (SO_RandUni() < tauF) ? (SO_RandBetween(Fl, Fl+Fu)) : (F[j]);
			SO_TElm newCR = (SO_RandUni() < tauCR) ? (SO_RandBetween(CRl, CRl+CRu)) : (CR[j]);

			/* Initialize crossover-weight vector. Vector-based. */
			SO_InitVector(w, newF, n);

			/* Store old position. */
			SO_CopyVector(t, x, n);

			/* Reset the random-set used for picking distinct agents.
			 * Exclude the j'th agent (also referred to as x). */
			RO_RandSetResetExclude(&randSet, j);

			/* Perform DE crossover. */
			SO_DEEngineCrossover(SO_kJDECrossover, n, newCR, w, x, g, numAgents, agents, &randSet);

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

				/* JDE 'Self-adaptive' parameters.
				 * Keep the new parameters because they led to fitness improvement. */
				F[j] = newF;
				CR[j] = newCR;
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
	SO_FreeVector(F);
	SO_FreeVector(CR);

	/* Free random set */
	RO_RandSetFree(&randSet);

	return gFitness;
}

/* ---------------------------------------------------------------- */
