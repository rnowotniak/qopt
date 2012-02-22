/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	DEEngine
 *
 *	See header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/DEEngine.h>
#include <SwarmOps/Tools/Random.h>
#include <RandomOps/RandomSet.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

/* Crossover for different DE variants. */
void SO_DEEngineCrossover(	const size_t variant,			/* Which DE variant to use. */
							SO_TDim n,						/* Dimensionality of problem. */
							const SO_TElm CR,				/* Crossover probability. */
							const SO_TElm *F,				/* Differential weight. */
							SO_TElm *x,						/* Destination vector. */
							const SO_TElm *g,				/* Best vector. */
							const size_t numAgents,			/* Number of agents. */
							const SO_TElm **agents,			/* All agent vectors. */
							struct RO_RandSet *randSet)		/* For picking random and distinct agents. */
{
	switch (variant)
	{
	case SO_kDECrossoverBest1Bin :
		SO_DEEngineBest1Bin(n, CR, F, x, g, numAgents, agents, randSet); break;

	case SO_kDECrossoverBest1BinSimple :
		SO_DEEngineBest1BinSimple(n, CR, F, x, g, numAgents, agents, randSet); break;

	case SO_kDECrossoverRand1BinEO :
		SO_DEEngineRand1BinEO(n, CR, F, x, numAgents, agents, randSet); break;

	case SO_kDECrossoverRand1Bin :
	default :
		SO_DEEngineRand1Bin(n, CR, F, x, numAgents, agents, randSet); break;
	}
}

/* ---------------------------------------------------------------- */

/* Crossover for DE/Best/1/Bin Variant. */
void SO_DEEngineBest1Bin(	SO_TDim n,					/* Dimensionality of problem. */
						const SO_TElm CR,				/* Crossover probability. */
						const SO_TElm *F,				/* Differential weight. */
						SO_TElm *x,						/* Destination vector. */
						const SO_TElm *g,				/* Best vector. */
						const size_t numAgents,			/* Number of agents. */
						const SO_TElm **agents,			/* All agent vectors. */
						struct RO_RandSet *randSet)		/* For picking random and distinct agents. */
{
	/* Iteration variable. */
	size_t k;

	/* Pick random dimension. */
	SO_TDim R = SO_RandIndex(n);

	/* Pick random and distinct agent-indices. */
	size_t R1 = RO_RandSetDraw(randSet, &SO_RandIndex);
	size_t R2 = RO_RandSetDraw(randSet, &SO_RandIndex);

	/* Refer to the randomly picked agents as a and b. */
	const SO_TElm *a = agents[R1];
	const SO_TElm *b = agents[R2];

	/* Compute potentially new position. */
	for (k=0; k<n; k++)
	{
		if (SO_RandUni()<CR || k==R)
		{
			x[k] = g[k] + F[k] * (a[k] - b[k]);
		}
	}
}

/* ---------------------------------------------------------------- */

/* Crossover for DE/Best/1/Bin/Simple Variant. */
void SO_DEEngineBest1BinSimple(
						SO_TDim n,						/* Dimensionality of problem. */
						const SO_TElm CR,				/* Crossover probability. */
						const SO_TElm *F,				/* Differential weight. */
						SO_TElm *x,						/* Destination vector. */
						const SO_TElm *g,				/* Best vector. */
						const size_t numAgents,			/* Number of agents. */
						const SO_TElm **agents,			/* All agent vectors. */
						struct RO_RandSet *randSet)		/* For picking random and distinct agents. */
{
	/* Iteration variable. */
	size_t k;

	/* Pick random dimension. */
	SO_TDim R = SO_RandIndex(n);

	/* Other agents to be picked at random. */
	const SO_TElm *a, *b;

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
		if (SO_RandUni()<CR || k==R)
		{
			x[k] = g[k] + F[k] * (a[k] - b[k]);
		}
	}
}

/* ---------------------------------------------------------------- */

/* Crossover for DE/Rand/1/Bin Variant. */
void SO_DEEngineRand1Bin(	SO_TDim n,					/* Dimensionality of problem. */
						const SO_TElm CR,				/* Crossover probability. */
						const SO_TElm *F,				/* Differential weight. */
						SO_TElm *x,						/* Destination vector. */
						const size_t numAgents,			/* Number of agents. */
						const SO_TElm **agents,			/* All agent vectors. */
						struct RO_RandSet *randSet)		/* For picking random and distinct agents. */
{
	/* Iteration variable. */
	size_t k;

	/* Pick random dimension. */
	SO_TDim R = SO_RandIndex(n);

	/* Pick random and distinct agent-indices. */
	size_t R1 = RO_RandSetDraw(randSet, &SO_RandIndex);
	size_t R2 = RO_RandSetDraw(randSet, &SO_RandIndex);
	size_t R3 = RO_RandSetDraw(randSet, &SO_RandIndex);

	/* Refer to the randomly picked agents as a and b. */
	const SO_TElm *a = agents[R1];
	const SO_TElm *b = agents[R2];
	const SO_TElm *c = agents[R3];

	/* Compute potentially new position. */
	for (k=0; k<n; k++)
	{
		if (SO_RandUni()<CR || k==R)
		{
			x[k] = c[k] + F[k] * (a[k] - b[k]);
		}
	}
}

/* ---------------------------------------------------------------- */

/* Crossover for DE/Rand/1/Bin Either-Or Variant. */
void SO_DEEngineRand1BinEO(	SO_TDim n,					/* Dimensionality of problem. */
						const SO_TElm CR,				/* Crossover probability. */
						const SO_TElm *F,				/* Differential weight. */
						SO_TElm *x,						/* Destination vector. */
						const size_t numAgents,			/* Number of agents. */
						const SO_TElm **agents,			/* All agent vectors. */
						struct RO_RandSet *randSet)		/* For picking random and distinct agents. */
{
	/* Iteration variable. */
	size_t k;

	/* Pick random dimension. */
	SO_TDim R = SO_RandIndex(n);

	/* Pick random and distinct agent-indices. */
	size_t R1 = RO_RandSetDraw(randSet, &SO_RandIndex);
	size_t R2 = RO_RandSetDraw(randSet, &SO_RandIndex);
	size_t R3 = RO_RandSetDraw(randSet, &SO_RandIndex);

	/* Refer to the randomly picked agents as a and b. */
	const SO_TElm *a = agents[R1];
	const SO_TElm *b = agents[R2];
	const SO_TElm *c = agents[R3];

	/* Compute potentially new position. */
	for (k=0; k<n; k++)
	{
		if (SO_RandBool())
		{
			if (SO_RandUni()<CR || k==R)
			{
				x[k] = c[k] + F[k] * (a[k] - b[k]);
			}
		}
		else
		{
			if (SO_RandUni()<CR || k==R)
			{
				x[k] = c[k] + 0.5 * (F[k]+1) * (a[k] + b[k] - 2*c[k]);
			}
		}
	}
}

/* ---------------------------------------------------------------- */
