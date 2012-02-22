/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	BenchmarkContext
 *
 *	The context-struct used for holding specifications for a
 *	benchmark problem.
 *
 *	Examples of benchmark problems are: Sphere, Rosenbrock, etc.
 *
 * ================================================================ */

#ifndef SO_BENCHMARKCONTEXT_H
#define SO_BENCHMARKCONTEXT_H

#include <SwarmOps/Tools/Types.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* The context-struct for benchmark problems. */
	struct SO_BenchmarkContext
	{
		SO_TDim		n;					/* Dimensionality. */
		int			displaceOptimum;	/* Displace optimum. */
	};

	/* ---------------------------------------------------------------- */

	/* Make and return a context-struct given the dimensionality. */
	struct SO_BenchmarkContext
		SO_MakeBenchmarkContext
		(SO_TDim n, int displaceOptimum);

	/* ---------------------------------------------------------------- */

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_BENCHMARKCONTEXT_H */

/* ================================================================ */
