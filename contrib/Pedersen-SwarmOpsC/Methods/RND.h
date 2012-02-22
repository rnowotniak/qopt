/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	RND
 *
 *	Random Sampling (RND). Positions from the search-space
 *	are sampled randomly and uniformly and the position with
 *	the best fitness is returned.
 *
 * ================================================================ */

#ifndef SO_RND_H
#define SO_RND_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Behavioural parameters and boundaries. */
#define SO_kNumParametersRND 0
#define SO_kParametersDefaultRND 0
#define SO_kParametersLowerRND 0
#define SO_kParametersUpperRND 0

	/* String containing the name of the optimization method. */
	extern const char SO_kNameRND[];

	/* Array of strings containing the parameter names of the optimization method. */
#define SO_kParameterNameRND 0

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_RND(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit);

	/* ---------------------------------------------------------------- */

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_RND_H */

/* ================================================================ */
