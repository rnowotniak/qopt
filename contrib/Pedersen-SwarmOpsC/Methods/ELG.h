/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	ELG
 *
 *	Evolution by Lingering Global best (ELG) optimization method
 *	derived as a simplification to the DE method.
 *
 * ================================================================ */

#ifndef SO_ELG_H
#define SO_ELG_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Behavioural parameters and boundaries. */

#define SO_kNumParametersELG 1

	extern const SO_TElm SO_kParametersDefaultELG[];
	extern const SO_TElm SO_kParametersLowerELG[];
	extern const SO_TElm SO_kParametersUpperELG[];

	/* String containing the name of the optimization method. */
	extern const char SO_kNameELG[];

	/* Array of strings containing the parameter names of the optimization method. */
	extern const char* SO_kParameterNameELG[];

	/* ---------------------------------------------------------------- */

	/* Functions for retrieving the individual parameters from a vector. */
	size_t SO_ELGNumAgents(SO_TElm const* param);

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_ELG(const SO_TElm *param, void const* context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_ELG_H */
