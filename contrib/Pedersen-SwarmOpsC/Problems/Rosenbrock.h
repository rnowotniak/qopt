/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Rosenbrock
 *
 *	The Rosenbrock benchmark optimization problem.
 *
 * ================================================================ */

#ifndef SO_ROSENBROCK_H
#define SO_ROSENBROCK_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Initialization and search-space boundaries. */
#define SO_kRosenbrockLowerInit 15
#define SO_kRosenbrockUpperInit 30
#define SO_kRosenbrockLowerBound -100
#define SO_kRosenbrockUpperBound 100
#define SO_kRosenbrockDisplace 25

	/* ---------------------------------------------------------------- */

	/* A string holding the problem's name. */
	extern const char SO_kNameRosenbrock[];

	/* ---------------------------------------------------------------- */

	/* The fitness function. */
	SO_TFitness SO_Rosenbrock(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit);

	/* The gradient of the fitness function. */
	SO_TDim SO_RosenbrockGradient(const SO_TElm *x, SO_TElm *v, void *context);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_ROSENBROCK_H */
