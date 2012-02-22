/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	GD
 *
 *	Gradient Descent (GD) optimization method.
 *	Note that it requires the gradient of the problem to
 *	be optimized.
 *
 * ================================================================ */

#ifndef SO_GD_H
#define SO_GD_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Behavioural parameters and their boundaries. */

#define SO_kNumParametersGD 1

	extern const SO_TElm SO_kParametersDefaultGD[];
	extern const SO_TElm SO_kParametersLowerGD[];
	extern const SO_TElm SO_kParametersUpperGD[];

	/* String containing the name of the optimization method. */
	extern const char SO_kNameGD[];

	/* Array of strings containing the parameter names of the optimization method. */
	extern const char* SO_kParameterNameGD[];

	/* ---------------------------------------------------------------- */

	/* Functions for retrieving the individual parameters from a vector. */
	SO_TElm SO_GDAlpha(const SO_TElm *param);

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_GD(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit);

	/* ---------------------------------------------------------------- */

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_GD_H */

/* ================================================================ */
