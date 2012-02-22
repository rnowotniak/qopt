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
 *	Differential Evolution (DE) optimization method originally
 *	due to Storner and Price. This suite offers combinations
 *	of DE variants and various perturbation schemes for its
 *	behavioural parameters. Use SO_kDESuiteCrossover and
 *	SO_kDESuiteDither to toggle between these. Note that this
 *	has complicated the implementation somewhat.
 *
 * ================================================================ */

#ifndef SO_DESuite_H
#define SO_DESuite_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <SwarmOps/Methods/DEEngine.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */
	/* Different configuration options. */

	/* Dither type. */
#define SO_kDESuiteDitherNone 0				/* No dither. */
#define SO_kDESuiteDitherGeneration 1		/* Generation based. */
#define SO_kDESuiteDitherVector 2			/* Vector based. */
#define SO_kDESuiteDitherElement 3			/* Vector-element based, aka. Jitter. */

	/* Set configuration to be used. (See also DEEngine.h) */
#define SO_kDESuiteCrossover SO_kDECrossoverBest1BinSimple
#define SO_kDESuiteDither SO_kDESuiteDitherNone

	/* ---------------------------------------------------------------- */
	/* Behavioural parameters and their boundaries. */

#if (SO_kDESuiteDither == SO_kDESuiteDitherNone)
#define SO_kNumParametersDESuite 3
#else
#define SO_kNumParametersDESuite 4
#endif

	extern const SO_TElm SO_kParametersDefaultDESuite[];
	extern const SO_TElm SO_kParametersLowerDESuite[];
	extern const SO_TElm SO_kParametersUpperDESuite[];

	/* String containing the name of the optimization method. */
	extern const char SO_kNameDESuite[];

	/* Array of strings containing the parameter names of the optimization method. */
	extern const char* SO_kParameterNameDESuite[];

	/* ---------------------------------------------------------------- */

	/* Functions for retrieving the individual parameters from a vector. */

	/* Number of agents. */
	size_t SO_DESuiteNumAgents(SO_TElm const* param);

	/* Crossover probability. */
	SO_TElm SO_DESuiteCrossoverP(SO_TElm const* param);

#if (SO_kDESuiteDither == SO_kDESuiteDitherNone)
	/* Differential weight. */
	SO_TElm SO_DESuiteF(SO_TElm const* param);
#else
	/* Differential weight midvalue. */
	SO_TElm SO_DESuiteFMid(SO_TElm const* param);

	/* Differential weight range. */
	SO_TElm SO_DESuiteFRange(SO_TElm const* param);
#endif

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_DESuite(const SO_TElm *param, void const* context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_DESuite_H */
