/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Sphere
 *
 *	The Sphere benchmark optimization problem.
 *
 * ================================================================ */

#ifndef SO_SPHERE_H
#define SO_SPHERE_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Initialization and search-space boundaries. */
#define SO_kSphereLowerInit 50
#define SO_kSphereUpperInit 100
#define SO_kSphereLowerBound -100
#define SO_kSphereUpperBound 100
#define SO_kSphereDisplace 25

	/*----------------------------------------------------------------*/

	/* A string holding the problem's name. */
	extern const char SO_kNameSphere[];

	/* ---------------------------------------------------------------- */

	/* The fitness function. */
	SO_TFitness SO_Sphere(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit);

	/* The gradient of the fitness function. */
	SO_TDim SO_SphereGradient(const SO_TElm *x, SO_TElm *v, void *context);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_SPHERE_H */
