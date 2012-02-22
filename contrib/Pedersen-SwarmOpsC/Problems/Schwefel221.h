/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Schwefel221
 *
 *	The Schwefel 2.21 benchmark optimization problem.
 *
 * ================================================================ */

#ifndef SO_SCHWEFEL221_H
#define SO_SCHWEFEL221_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Initialization and search-space boundaries. */
#define SO_kSchwefel221LowerInit 50
#define SO_kSchwefel221UpperInit 100
#define SO_kSchwefel221LowerBound -100
#define SO_kSchwefel221UpperBound 100
#define SO_kSchwefel221Displace -25

	/*----------------------------------------------------------------*/

	/* A string holding the problem's name. */
	extern const char SO_kNameSchwefel221[];

	/* ---------------------------------------------------------------- */

	/* The fitness function. */
	SO_TFitness SO_Schwefel221(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit);

	/* The gradient of the fitness function. */
	SO_TDim SO_Schwefel221Gradient(const SO_TElm *x, SO_TElm *v, void *context);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_SCHWEFEL221_H */
