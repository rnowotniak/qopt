/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Types
 *
 *	Type-definitions and related constant values.
 *
 * ================================================================ */

#ifndef SO_TYPES_H
#define SO_TYPES_H

#include <stddef.h>
#include <float.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Fitness-type. */
	typedef double SO_TFitness;

	/* Maximum (worst) possible fitness value. */
#define SO_kFitnessMax DBL_MAX

	/* Minimum (worst) possible fitness value. Normally you would use
	 * a value of zero instead, as this is really the range of the datatype.
	 * SwarmOps assumes fitnesses are non-negative for use in meta-optimization. */
#define SO_kFitnessMin DBL_MIN

	/* ---------------------------------------------------------------- */

	/* Dimensionality of an optimization problem. */
	typedef size_t SO_TDim;

	/* Element of a vector in a search-space. */
	typedef double SO_TElm;

	/* ---------------------------------------------------------------- */

	/* Type for a fitness function. That is, an optimization problem. */
	typedef SO_TFitness (*SO_FProblem) (const SO_TElm *x, void *context, const SO_TFitness fitnessLimit);

	/* Type for a gradient function for an optimization problem. */
	typedef SO_TDim (*SO_FGradient) (const SO_TElm *x, SO_TElm *v, void *context);

	/* Type for an optimization method. This is identical to the type for
	 * an optimization problem, which makes optimizing the behavioural parameters
	 * for an optimization method possible. */
	typedef SO_FProblem SO_FMethod;

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_TYPES_H */
