/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	MYG
 *
 *	More Yo-yos doing Global optimization (MYG) devised as a
 *	simplification to the DE optimization method originally due
 *	to Storner and Price. The MYG method eliminates the probability
 *	parameter, and also has random selection of which agent to
 *	update instead of iterating over them all in order.
 *
 * ================================================================ */

#ifndef SO_MYG_H
#define SO_MYG_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Behavioural parameters and their boundaries. */

#define SO_kNumParametersMYG 2

	extern const SO_TElm SO_kParametersDefaultMYG[];
	extern const SO_TElm SO_kParametersLowerMYG[];
	extern const SO_TElm SO_kParametersUpperMYG[];

	/* String containing the name of the optimization method. */
	extern const char SO_kNameMYG[];

	/* Array of strings containing the parameter names of the optimization method. */
	extern const char* SO_kParameterNameMYG[];

	/* ---------------------------------------------------------------- */

	/* Functions for retrieving the individual parameters from a vector. */

	/* Number of agents. */
	size_t SO_MYGNumAgents(SO_TElm const* param);

	/* Differential weight. */
	SO_TElm SO_MYGF(SO_TElm const* param);

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_MYG(const SO_TElm *param, void const* context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_MYG_H */
