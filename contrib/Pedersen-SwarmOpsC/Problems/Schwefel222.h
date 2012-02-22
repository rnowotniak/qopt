/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Schwefel222
 *
 *	The Schwefel 2.22 benchmark optimization problem.
 *
 * ================================================================ */

#ifndef SO_SCHWEFEL222_H
#define SO_SCHWEFEL222_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Initialization and search-space boundaries. */
#define SO_kSchwefel222LowerInit 5
#define SO_kSchwefel222UpperInit 10
#define SO_kSchwefel222LowerBound -10
#define SO_kSchwefel222UpperBound 10
#define SO_kSchwefel222Displace -2.5

	/*----------------------------------------------------------------*/

	/* A string holding the problem's name. */
	extern const char SO_kNameSchwefel222[];

	/* ---------------------------------------------------------------- */

	/* The fitness function. */
	SO_TFitness SO_Schwefel222(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit);

	/* The gradient of the fitness function. */
	SO_TDim SO_Schwefel222Gradient(const SO_TElm *x, SO_TElm *v, void *context);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_SCHWEFEL222_H */
