/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Griewank
 *
 *	The Griewank benchmark optimization problem.
 *
 * ================================================================ */

#ifndef SO_GRIEWANK_H
#define SO_GRIEWANK_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Initialization and search-space boundaries. */
#define SO_kGriewankLowerInit 300
#define SO_kGriewankUpperInit 600
#define SO_kGriewankLowerBound -600
#define SO_kGriewankUpperBound 600
#define SO_kGriewankDisplace -150

	/* ---------------------------------------------------------------- */

	/* A string holding the problem's name. */
	extern const char SO_kNameGriewank[];

	/* ---------------------------------------------------------------- */

	/* The fitness function. */
	SO_TFitness SO_Griewank(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit);

	/* The gradient of the fitness function. */
	SO_TDim SO_GriewankGradient(const SO_TElm *x, SO_TElm *v, void *context);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_GRIEWANK_H */
