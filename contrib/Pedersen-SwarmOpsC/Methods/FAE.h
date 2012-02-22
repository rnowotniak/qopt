/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	FAE
 *
 *	Forever Accumulating Evolution (DE) optimization method derived
 *	as a simplification to the PSO method.
 *
 * ================================================================ */

#ifndef SO_FAE_H
#define SO_FAE_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Behavioural parameters and boundaries. */

#define SO_kNumParametersFAE 3

	extern const SO_TElm SO_kParametersDefaultFAE[];
	extern const SO_TElm SO_kParametersLowerFAE[];
	extern const SO_TElm SO_kParametersUpperFAE[];

	/* String containing the name of the optimization method. */
	extern const char SO_kNameFAE[];

	/* Array of strings containing the parameter names of the optimization method. */
	extern const char* SO_kParameterNameFAE[];

	/* ---------------------------------------------------------------- */

	/* Functions for retrieving the individual parameters from a vector. */
	size_t SO_FAENumAgents(SO_TElm const* param);
	SO_TElm SO_FAELambdaG(SO_TElm const* param);
	SO_TElm SO_FAELambdaX(SO_TElm const* param);

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_FAE(const SO_TElm *param, void const* context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_FAE_H */
