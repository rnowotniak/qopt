/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Methods
 *
 *	A compilation of all available optimization methods and
 *	various data for them.
 *
 * ================================================================ */

#ifndef SO_METHODS_H
#define SO_METHODS_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Number of optimization methods available. */
#define SO_kNumMethods 18

	/* Identification handles for all optimization methods. */
	enum
	{
		/* Mesh Iteration. */
		SO_kMethodMESH,				/* Mesh iteration. */

		/* Random Sampling */
		SO_kMethodRND,				/* Random Sampling (Uniform) */

		/* Gradient-Based Optimization */
		SO_kMethodGD,				/* Gradient Descent */
		SO_kMethodGED,				/* Gradient Emancipated Descent */

		/* Local Sampling */
		SO_kMethodHC,				/* Hill-Climber */
		SO_kMethodSA,				/* Simulated Annealing */
		SO_kMethodPS,				/* Pattern Search */
		SO_kMethodLUS,				/* Local Unimodal Sampling */

		/* Swarm-Based Optimization, DE and variants */
		SO_kMethodDE,				/* Differential Evolution */
		SO_kMethodDESuite,			/* Differential Evolution Suite */
		SO_kMethodDETP,				/* DE with Temporal Parameters */
		SO_kMethodJDE,				/* Jan. Differential Evolution (jDE) */
		SO_kMethodELG,				/* Evolution by Lingering Global best */
		SO_kMethodMYG,				/* More Yo-yos doing Global optimization */

		/* Swarm-Based Optimization, PSO and variants */
		SO_kMethodPSO,				/* Particle Swarm Optimization */
		SO_kMethodFAE,				/* Forever Accumulating Evolution */
		SO_kMethodMOL,				/* Many Optimizing Liaisons */

		/* Compound Methods */
		SO_kMethodLICE,				/* Layered and Interleaved Co-Evolution */
	};

	/* ---------------------------------------------------------------- */

	/* Array of functions for the optimization methods. */
	extern const SO_FMethod SO_kMethods[];

	/* Array of names for the optimization methods. */
	extern const char* SO_kMethodName[];

	/* Array of the number of parameters for each optimization method. */
	extern const size_t SO_kMethodNumParameters[];

	/* Array-of-arrays holding parameter names for the optimization methods. */
	extern const char** SO_kMethodParameterName[];

	/* Array of default parameters for the optimization methods. */
	extern const SO_TElm* SO_kMethodDefaultParameters[];

	/* Arrays of initialization and search-space boundaries for the
	 * behavioural parameters of the optimization methods. */
	extern const SO_TElm* SO_kMethodLowerInit[];
	extern const SO_TElm* SO_kMethodUpperInit[];
	extern const SO_TElm* SO_kMethodLowerBound[];
	extern const SO_TElm* SO_kMethodUpperBound[];

	/*----------------------------------------------------------------*/

	/* Return whether an optimization method requires the gradient
	 * of the problem to be optimized. This is e.g. true for the GD
	 * and GED methods. */
	int SO_RequiresGradient(size_t methodId);

	/*----------------------------------------------------------------*/

	/* Print the given parameters and their associated names. */
	void SO_PrintParameters(const size_t methodId, SO_TElm const* param);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_METHODS_H */
