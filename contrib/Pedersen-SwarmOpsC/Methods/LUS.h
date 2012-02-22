/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2007 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	LUS
 *
 *	Local Unimodal Sampling (LUS). Does local sampling
 *	with an exponential decrease of the sampling-range.
 *
 * ================================================================ */

#ifndef SO_LUS_H
#define SO_LUS_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Behavioural parameters and their boundaries. */

#define SO_kNumParametersLUS 1

	extern const SO_TElm SO_kParametersDefaultLUS[];
	extern const SO_TElm SO_kParametersLowerLUS[];
	extern const SO_TElm SO_kParametersUpperLUS[];

	/* String containing the name of the optimization method. */
	extern const char SO_kNameLUS[];

	/* Array of strings containing the parameter names of the optimization method. */
	extern const char* SO_kParameterNameLUS[];

	/* ---------------------------------------------------------------- */

	/* Functions for retrieving the individual parameters from a vector. */

	/* The decrease-factor. */
	SO_TElm SO_LUSGamma(const SO_TElm *param);

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_LUS(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit);

	/* ---------------------------------------------------------------- */

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_LUS_H */

/* ================================================================ */
