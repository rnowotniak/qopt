/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Ackley
 *
 *	The Ackley benchmark optimization problem.
 *
 * ================================================================ */

#ifndef SO_ACKLEY_H
#define SO_ACKLEY_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Initialization and search-space boundaries. */
#define SO_kAckleyLowerInit 15
#define SO_kAckleyUpperInit 30
#define SO_kAckleyLowerBound -30
#define SO_kAckleyUpperBound 30
#define SO_kAckleyDisplace -7.5

	/* ---------------------------------------------------------------- */

	/* A string holding the problem's name. */
	extern const char SO_kNameAckley[];

	/* ---------------------------------------------------------------- */

	/* The fitness function. */
	SO_TFitness SO_Ackley(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit);

	/* The gradient of the fitness function. */
	SO_TDim SO_AckleyGradient(const SO_TElm *x, SO_TElm *v, void *context);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_ACKLEY_H */
