/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	DE
 *
 *	Differential Evolution (DE) optimization method originally
 *	due to Storner and Price.
 *
 * ================================================================ */

#ifndef SO_DE_H
#define SO_DE_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Behavioural parameters and their boundaries. */

#define SO_kNumParametersDE 3

	extern const SO_TElm SO_kParametersDefaultDE[];
	extern const SO_TElm SO_kParametersLowerDE[];
	extern const SO_TElm SO_kParametersUpperDE[];

	/* String containing the name of the optimization method. */
	extern const char SO_kNameDE[];

	/* Array of strings containing the parameter names of the optimization method. */
	extern const char* SO_kParameterNameDE[];

	/* ---------------------------------------------------------------- */

	/* Functions for retrieving the individual parameters from a vector. */

	/* Number of agents NP. */
	size_t SO_DENumAgents(SO_TElm const* param);

	/* Crossover probability CR. */
	SO_TElm SO_DECR(SO_TElm const* param);

	/* Differential weight F. */
	SO_TElm SO_DEF(SO_TElm const* param);

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_DE(const SO_TElm *param, void const* context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_DE_H */
