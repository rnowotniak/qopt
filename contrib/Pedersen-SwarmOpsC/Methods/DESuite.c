/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	DESuite
 *
 *	See header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/DESuite.h>
#include <SwarmOps/Tools/Bound.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Matrix.h>
#include <SwarmOps/Tools/Init.h>
#include <SwarmOps/Tools/Random.h>
#include <RandomOps/RandomSet.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

/* Determine crossover-name. */
#if (SO_kDESuiteCrossover == SO_kDECrossoverRand1Bin)
#define SO_kNameDESuiteCrossover SO_kNameDECrossoverRand1Bin
#elif (SO_kDESuiteCrossover == SO_kDECrossoverRand1BinEO)
#define SO_kNameDESuiteCrossover SO_kNameDECrossoverRand1BinEO
#elif (SO_kDESuiteCrossover == SO_kDECrossoverBest1Bin)
#define SO_kNameDESuiteCrossover SO_kNameDECrossoverBest1Bin
#elif (SO_kDESuiteCrossover == SO_kDECrossoverBest1BinSimple)
#define SO_kNameDESuiteCrossover SO_kNameDECrossoverBest1BinSimple
#endif

/* Determine dither-name. */
#if (SO_kDESuiteDither == SO_kDESuiteDitherNone)
#define SO_kNameDESuiteDither ""
#elif (SO_kDESuiteDither == SO_kDESuiteDitherGeneration)
#define SO_kNameDESuiteDither "GenDither"
#elif (SO_kDESuiteDither == SO_kDESuiteDitherVector)
#define SO_kNameDESuiteDither "VecDither"
#elif (SO_kDESuiteDither == SO_kDESuiteDitherElement)
#define SO_kNameDESuiteDither "Jitter"
#endif

const char SO_kNameDESuite[] = "DE" SO_kNameDESuiteCrossover SO_kNameDESuiteDither;

/* ---------------------------------------------------------------- */

#if (SO_kDESuiteDither == SO_kDESuiteDitherNone)

/* The method has the following parameters:
 * Index 0: Number of agents
 * Index 1: Crossover probability (CR)
 * Index 2: Differential weight (F) (note different order from DE (Basic))
 */

/* Parameter names. */
const char* SO_kParameterNameDESuite[] = {"NP", "CR", "F"};

/* Parameter boundaries. */
const SO_TElm SO_kParametersLowerDESuite[SO_kNumParametersDESuite] = {4, 0, 0};
const SO_TElm SO_kParametersUpperDESuite[SO_kNumParametersDESuite] = {200, 1, 2.0};

/* Parameters for various DE configurations. */
#if (SO_kDESuiteCrossover == SO_kDECrossoverBest1Bin)
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {157.0, 0.976920, 0.334942};
#elif (SO_kDESuiteCrossover == SO_kDECrossoverBest1BinSimple)
/* Tuned for 3 benchmark problems, 200*dim runlength. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {157.0, 0.976920, 0.334942};
#elif (SO_kDESuiteCrossover == SO_kDECrossoverRand1Bin)

#if 0
/* Tuned for 12 benchmark problems, 200*dim runlength. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {10.0, 0.031855, 0.733094};
#elif 0
/* Tuned for 3 benchmark problems, 200*dim runlength. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {8.0, 0.131305, 0.776182};
#elif 1
/* Tuned for ANN Cancer problem, 20*dim runlength. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {9, 0.804681, 0.736314};
#else
/* Default parameters. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {300.0, 0.9, 0.5};
#endif

#else
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {44, 0.967665, 0.536893};
#endif

#else

/* The method has the following parameters:
 * Index 0: Number of agents
 * Index 1: Crossover probability
 * Index 2: Differential weight mid-value (FMid))
 * Index 3: Differential weight range (FRange))
 */

/* Parameter names. */
const char* SO_kParameterNameDESuite[] = {"NP", "CR", "F_{mid}", "F_{range}"};

/* Parameter boundaries. */
const SO_TElm SO_kParametersLowerDESuite[SO_kNumParametersDESuite] = {4, 0, 0, 0};
const SO_TElm SO_kParametersUpperDESuite[SO_kNumParametersDESuite] = {200, 1, 2.0, 3.0};

/* Default parameters for various DE configurations. */
#if (SO_kDESuiteCrossover == SO_kDECrossoverRand1Bin && SO_kDESuiteDither == SO_kDESuiteDitherVector)

#if 0
/* Tuned for 12 benchmark problems, 200*dim runlength. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {7.0, 0.021481, 0.849680, 1.779813};
#elif 0
/* Tuned for 3 benchmark problems, 200*dim runlength. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {7.0, 0.001779, 1.203716, 1.931654};
#elif 1
/* Tuned for ANN Cancer problem, 20*dim runlength. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {9.0, 0.824410, 0.833241, 1.553993};
#else
/* Default parameters. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {300.0, 0.9, 0.75, 0.25};
#endif

#elif (SO_kDESuiteCrossover == SO_kDECrossoverRand1Bin && SO_kDESuiteDither == SO_kDESuiteDitherElement)

#if 0
/* Tuned for 12 benchmark problems, 200*dim runlength. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {11.0, 0.096154, 0.503464, 0.954235};
#elif 0
/* Tuned for 3 benchmark problems, 200*dim runlength. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {7.0, 0.170015, 0.788930, 0.598157};
#elif 1
/* Tuned for ANN Cancer problem, 20*dim runlength. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {7.0, 0.916751, 0.809399, 0.716685};
#else
/* Default parameters. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {300.0, 0.9, 0.5, 0.0005};
#endif

#elif (SO_kDESuiteCrossover == SO_kDECrossoverBest1Bin && SO_kDESuiteDither == SO_kDESuiteDitherGeneration)
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {138.0, 0.969859, 0.492406, 0.282430};
#elif (SO_kDESuiteCrossover == SO_kDECrossoverBest1Bin && SO_kDESuiteDither == SO_kDESuiteDitherVector)
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {17, 0.388487, 0.269755, 1.818551};
#elif (SO_kDESuiteCrossover == SO_kDECrossoverBest1Bin && SO_kDESuiteDither == SO_kDESuiteDitherElement)
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {7, 0.003339, 0.822213, 1.887380};
#elif (SO_kDESuiteCrossover == SO_kDECrossoverBest1BinSimple && SO_kDESuiteDither == SO_kDESuiteDitherVector)

# if 1
/* Tuned for ANN Cancer problem, 20*dim runlength. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {47.0, 0.954343, 0.391178, 0.843602};
#else
/* Tuned for 3 benchmark problems, 200*dim runlength. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {183.0, 0.936300, 0.321421, 0.480854};
#endif

#elif (SO_kDESuiteCrossover == SO_kDECrossoverBest1BinSimple && SO_kDESuiteDither == SO_kDESuiteDitherElement)

#if 1
/* Tuned for ANN Cancer problem, 20*dim runlength. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {13.0, 0.933635, 0.513652, 1.007801};
#else
/* Tuned for 3 benchmark problems, 200*dim runlength. */
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {5.0, 0.003546, 0.077658, 2.618717};
#endif

#else
const SO_TElm SO_kParametersDefaultDESuite[SO_kNumParametersDESuite] = {44, 0.967665, 0.536893, 0.1};
#endif

#endif

/* ---------------------------------------------------------------- */

size_t SO_DESuiteNumAgents(SO_TElm const* param)
{
	assert(param);

	return (size_t) (param[0]+0.5);
}

SO_TElm SO_DESuiteCrossoverP(SO_TElm const* param)
{
	assert(param);

	return param[1];
}

#if (SO_kDESuiteDither == SO_kDESuiteDitherNone)

SO_TElm SO_DESuiteF(SO_TElm const* param)
{
	assert(param);

	return param[2];
}

#else

SO_TElm SO_DESuiteFMid(SO_TElm const* param)
{
	assert(param);

	return param[2];
}

SO_TElm SO_DESuiteFRange(SO_TElm const* param)
{
	assert(param);

	return param[3];
}

#endif

/* ---------------------------------------------------------------- */

/* The overall structure of this function is:
 * - Retrieve variables from context and parameters.
 * - Allocate and initialize vectors and other data needed by this optimization method.
 * - Perform optimization.
 * - De-allocate data.
 * - Return best result from the optimization.
 */

SO_TFitness SO_DESuite(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit)
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
	size_t numAgents = SO_DESuiteNumAgents(param);
	SO_TElm CR = SO_DESuiteCrossoverP(param);

#if (SO_kDESuiteDither == SO_kDESuiteDitherNone)
	SO_TElm F = SO_DESuiteF(param);
#else
	SO_TElm FMid = SO_DESuiteFMid(param);
	SO_TElm FRange = SO_DESuiteFRange(param);
#endif

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

#if (SO_kDESuiteDither == SO_kDESuiteDitherNone)
	/* Initialize crossover-weight vector.
	 * Same value for all elements, vectors, and generations. */
	SO_InitVector(w, F, n);
#endif

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

#if (SO_kDESuiteDither == SO_kDESuiteDitherGeneration)
		/* Initialize crossover-weight vector. Generation-based. */
		SO_InitVector(w, SO_RandBetween(FMid-FRange, FMid+FRange), n);
#endif

		for (j=0; j<numAgents && i<numIterations; j++, i++)
		{
			/* Refer to the j'th agent as x. */
			SO_TElm* x = agents[j];

			/* Store old position. */
			SO_CopyVector(t, x, n);

			/* Reset the random-set used for picking distinct agents.
			 * Exclude the j'th agent (also referred to as x). */
			RO_RandSetResetExclude(&randSet, j);

#if (SO_kDESuiteDither == SO_kDESuiteDitherVector)
			/* Initialize crossover-weight vector. Vector-based. */
			SO_InitVector(w, SO_RandBetween(FMid-FRange, FMid+FRange), n);
#elif (SO_kDESuiteDither == SO_kDESuiteDitherElement)
			{
				size_t k;

				/* Initialize crossover-weight vector. Element-based. */
				for (k=0; k<n; k++)
				{
					w[k] = SO_RandBetween(FMid-FRange, FMid+FRange);
				}
			}
#endif

			/* Perform DE crossover. */
			SO_DEEngineCrossover(SO_kDESuiteCrossover, n, CR, w, x, g, numAgents, agents, &randSet);

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
