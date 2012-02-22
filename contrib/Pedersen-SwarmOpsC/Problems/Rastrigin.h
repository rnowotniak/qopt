/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Rastrigin
 *
 *	The Rastrigin benchmark optimization problem.
 *
 * ================================================================ */

#ifndef SO_RASTRIGIN_H
#define SO_RASTRIGIN_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Initialization and search-space boundaries. */
#define SO_kRastriginLowerInit 2.56
#define SO_kRastriginUpperInit 5.12
#define SO_kRastriginLowerBound -5.12
#define SO_kRastriginUpperBound 5.12
#define SO_kRastriginDisplace 1.28

	/* ---------------------------------------------------------------- */

	/* A string holding the problem's name. */
	extern const char SO_kNameRastrigin[];

	/* ---------------------------------------------------------------- */

	/* The fitness function. */
	SO_TFitness SO_Rastrigin(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit);

	/* The gradient of the fitness function. */
	SO_TDim SO_RastriginGradient(const SO_TElm *x, SO_TElm *v, void *context);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_RASTRIGIN_H */
