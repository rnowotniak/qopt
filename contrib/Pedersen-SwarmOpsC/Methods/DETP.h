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
 *	Differential Evolution (DE) optimization method originally
 *	due to Storner and Price. This variant uses Temporal
 *	Parameters, that is, different parameters are used for
 *	different periods of the optimization run.
 *
 *	Use SO_kDETPCrossover to toggle between different
 *	crossover types, and SO_kDETPNumPeriods to set the
 *	number of temporal periods to use (1, 2, 4, and 8
 *	in default implementation, otherwise you need to
 *	modify the source-file).
 *
 * ================================================================ */

#ifndef SO_DETP_H
#define SO_DETP_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <SwarmOps/Methods/DEEngine.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Set DE configuration to be used (see DEEngine.h) */

#define SO_kDETPCrossover SO_kDECrossoverRand1Bin

	/* ---------------------------------------------------------------- */
	/* Behavioural parameters and their boundaries. */

#define SO_kDETPNumPeriods 2
#define SO_kNumParametersDETP (1+SO_kDETPNumPeriods*2)

	extern const SO_TElm SO_kParametersDefaultDETP[];
	extern const SO_TElm SO_kParametersLowerDETP[];
	extern const SO_TElm SO_kParametersUpperDETP[];

	/* String containing the name of the optimization method. */
	extern const char SO_kNameDETP[];

	/* Array of strings containing the parameter names of the optimization method. */
	extern const char* SO_kParameterNameDETP[];

	/* ---------------------------------------------------------------- */

	/* Functions for retrieving the individual parameters from a vector. */

	/* Number of agents. */
	size_t SO_DETPNumAgents(SO_TElm const* param);

	/* Crossover probability. The variable 'i' determines the current iteration,
	 * and kNumIterations is the total number of iterations to perform. */
	SO_TElm SO_DETPCrossoverP(SO_TElm const* param, const size_t i, const size_t kNumIterations);

	/* Differential weight. The variable 'i' determines the current iteration,
	 * and kNumIterations is the total number of iterations to perform. */
	SO_TElm SO_DETPF(SO_TElm const* param, const size_t i, const size_t kNumIterations);

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_DETP(const SO_TElm *param, void const* context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_DETP_H */
