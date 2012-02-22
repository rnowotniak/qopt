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
 *	Differential Evolution (DE) functinos for performing
 *	various kinds of crossover, etc.
 *
 * ================================================================ */

#ifndef SO_DEEngine_H
#define SO_DEEngine_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */
	/* Different configuration options. */

	/* Crossover variants. */
#define SO_kDECrossoverRand1Bin 0			/* DE/rand/1/bin */
#define SO_kDECrossoverRand1BinEO 1			/* DE/rand/1/bin Either-Or */
#define SO_kDECrossoverBest1Bin 2			/* DE/best/1/bin */
#define SO_kDECrossoverBest1BinSimple 3		/* DE/best/1/bin/Simple */

	/* Crossover names. */
#define SO_kNameDECrossoverRand1Bin "rand1bin"
#define SO_kNameDECrossoverRand1BinEO "rand1binEO"
#define SO_kNameDECrossoverBest1Bin "best1bin"
#define SO_kNameDECrossoverBest1BinSimple "best1binsimple"

	/* ---------------------------------------------------------------- */

	const char** SO_kNameDECrossover[];

	/* ---------------------------------------------------------------- */

	/* Crossover for different DE variants. */
	void SO_DEEngineCrossover(	const size_t variant,			/* Which DE variant to use. */
								SO_TDim n,						/* Dimensionality of problem. */
								const SO_TElm CR,				/* Crossover probability. */
								const SO_TElm *F,				/* Differential weight (vector). */
								SO_TElm *x,						/* Destination vector. */
								const SO_TElm *g,				/* Best vector. */
								const size_t numAgents,			/* Number of agents. */
								const SO_TElm **agents,			/* All agent vectors. */
								struct RO_RandSet *randSet);	/* For picking random and distinct agents. */

	/* ---------------------------------------------------------------- */

	/* Crossover for DE/Best/1/Bin Variant. */
	void SO_DEEngineBest1Bin(	SO_TDim n,						/* Dimensionality of problem. */
								const SO_TElm CR,				/* Crossover probability. */
								const SO_TElm *F,				/* Differential weight (vector). */
								SO_TElm *x,						/* Destination vector. */
								const SO_TElm *g,				/* Best vector. */
								const size_t numAgents,			/* Number of agents. */
								const SO_TElm **agents,			/* All agent vectors. */
								struct RO_RandSet *randSet);	/* For picking random and distinct agents. */

	/* ---------------------------------------------------------------- */

	/* Crossover for DE/Best/1/Bin/Simple Variant. */
	void SO_DEEngineBest1BinSimple(
								SO_TDim n,						/* Dimensionality of problem. */
								const SO_TElm CR,				/* Crossover probability. */
								const SO_TElm *F,				/* Differential weight (vector). */
								SO_TElm *x,						/* Destination vector. */
								const SO_TElm *g,				/* Best vector. */
								const size_t numAgents,			/* Number of agents. */
								const SO_TElm **agents,			/* All agent vectors. */
								struct RO_RandSet *randSet);	/* For picking random and distinct agents. */

	/* ---------------------------------------------------------------- */

	/* Crossover for DE/Rand/1/Bin Variant. */
	void SO_DEEngineRand1Bin(	SO_TDim n,						/* Dimensionality of problem. */
								const SO_TElm CR,				/* Crossover probability. */
								const SO_TElm *F,				/* Differential weight (vector). */
								SO_TElm *x,						/* Destination vector. */
								const size_t numAgents,			/* Number of agents. */
								const SO_TElm **agents,			/* All agent vectors. */
								struct RO_RandSet *randSet);	/* For picking random and distinct agents. */

	/* ---------------------------------------------------------------- */

	/* Crossover for DE/Rand/1/Bin Either-Or Variant. */
	void SO_DEEngineRand1BinEO(	SO_TDim n,						/* Dimensionality of problem. */
								const SO_TElm CR,				/* Crossover probability. */
								const SO_TElm *F,				/* Differential weight (vector). */
								SO_TElm *x,						/* Destination vector. */
								const size_t numAgents,			/* Number of agents. */
								const SO_TElm **agents,			/* All agent vectors. */
								struct RO_RandSet *randSet);	/* For picking random and distinct agents. */

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_DEEngine_H */
