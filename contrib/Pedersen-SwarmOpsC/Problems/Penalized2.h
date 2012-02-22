/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Penalized2
 *
 *	The Penalized2 benchmark optimization problem.
 *
 * ================================================================ */

#ifndef SO_PENALIZED2_H
#define SO_PENALIZED2_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Initialization and search-space boundaries. */
#define SO_kPenalized2LowerInit -5
#define SO_kPenalized2UpperInit 50
#define SO_kPenalized2LowerBound -50
#define SO_kPenalized2UpperBound 50

	/* Optimum displacement disabled for this problem because of penalty function.
	 * #define SO_kPenalized1Displace -12.5
	 */

	/*----------------------------------------------------------------*/

	/* A string holding the problem's name. */
	extern const char SO_kNamePenalized2[];

	/* ---------------------------------------------------------------- */

	/* The fitness function. */
	SO_TFitness SO_Penalized2(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit);

	/* The gradient of the fitness function. */
	SO_TDim SO_Penalized2Gradient(const SO_TElm *x, SO_TElm *v, void *context);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_PENALIZED2_H */
