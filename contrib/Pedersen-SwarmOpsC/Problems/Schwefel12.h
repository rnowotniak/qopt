/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Schwefel12
 *
 *	The Schwefel 1.2 benchmark optimization problem.
 *
 * ================================================================ */

#ifndef SO_SCHWEFEL12_H
#define SO_SCHWEFEL12_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Initialization and search-space boundaries. */
#define SO_kSchwefel12LowerInit 50
#define SO_kSchwefel12UpperInit 100
#define SO_kSchwefel12LowerBound -100
#define SO_kSchwefel12UpperBound 100
#define SO_kSchwefel12Displace -25

	/*----------------------------------------------------------------*/

	/* A string holding the problem's name. */
	extern const char SO_kNameSchwefel12[];

	/* ---------------------------------------------------------------- */

	/* The fitness function. */
	SO_TFitness SO_Schwefel12(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit);

	/* The gradient of the fitness function. */
	SO_TDim SO_Schwefel12Gradient(const SO_TElm *x, SO_TElm *v, void *context);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_SCHWEFEL12_H */
