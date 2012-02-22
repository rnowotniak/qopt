/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	GED
 *
 *	Gradient Emancipated Descent (GED) optimization method.
 *	Similar to basic GD, except that GED only evaluates the
 *	fitness according to a probability. This saves runtime.
 *
 * ================================================================ */

#ifndef SO_GED_H
#define SO_GED_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Behavioural parameters and their boundaries. */

#define SO_kNumParametersGED 2

	extern const SO_TElm SO_kParametersDefaultGED[];
	extern const SO_TElm SO_kParametersLowerGED[];
	extern const SO_TElm SO_kParametersUpperGED[];

	/* String containing the name of the optimization method. */
	extern const char SO_kNameGED[];

	/* Array of strings containing the parameter names of the optimization method. */
	extern const char* SO_kParameterNameGED[];

	/* ---------------------------------------------------------------- */

	/* Functions for retrieving the individual parameters from a vector. */
	SO_TElm SO_GEDAlpha(const SO_TElm *param);
	SO_TElm SO_GEDP(const SO_TElm *param);

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_GED(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit);

	/* ---------------------------------------------------------------- */

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_GED_H */

/* ================================================================ */
