/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Penalized1
 *
 *	The Penalized1 benchmark optimization problem.
 *
 * ================================================================ */

#ifndef SO_PENALIZED1_H
#define SO_PENALIZED1_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Initialization and search-space boundaries. */
#define SO_kPenalized1LowerInit 5
#define SO_kPenalized1UpperInit 50
#define SO_kPenalized1LowerBound -50
#define SO_kPenalized1UpperBound 50

	/* Optimum displacement disabled for this problem because of penalty function.
	 * #define SO_kPenalized1Displace 12.5
	 */

	/*----------------------------------------------------------------*/

	/* A string holding the problem's name. */
	extern const char SO_kNamePenalized1[];

	/* ---------------------------------------------------------------- */

	/* Penalty function, also used by Penalized2. */
	SO_TElm SO_PenalizedU(const SO_TElm x, const SO_TElm a, const SO_TElm k, const SO_TElm m);

	/* ---------------------------------------------------------------- */

	/* The fitness function. */
	SO_TFitness SO_Penalized1(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit);

	/* The gradient of the fitness function. */
	SO_TDim SO_Penalized1Gradient(const SO_TElm *x, SO_TElm *v, void *context);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_PENALIZED1_H */
