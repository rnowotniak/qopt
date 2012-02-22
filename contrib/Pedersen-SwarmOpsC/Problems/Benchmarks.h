/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Benchmarks
 *
 *	Contains all the benchmark optimization problems.
 *
 * ================================================================ */

#ifndef SO_BENCHMARK_H
#define SO_BENCHMARK_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* The number of benchmark problems available. */
#define SO_kNumBenchmark 12

	/* Identification numbers for the benchmark problems.
	 * These must start with zero and numbered incrementally.
	 * Ordering must match that in Benchmarks.c file. */
	enum
	{
		SO_kBenchmarkSphere,
		SO_kBenchmarkSchwefel222,
		SO_kBenchmarkSchwefel12,
		SO_kBenchmarkSchwefel221,
		SO_kBenchmarkGriewank,
		SO_kBenchmarkPenalized1,
		SO_kBenchmarkPenalized2,
		SO_kBenchmarkRastrigin,
		SO_kBenchmarkAckley,
		SO_kBenchmarkRosenbrock,
		SO_kBenchmarkStep,
		SO_kBenchmarkQuarticNoise
	};

	/* ---------------------------------------------------------------- */

	/* The benchmark fitness functions. */
	extern const SO_FProblem SO_kBenchmarkProblems[];

	/* The gradient-functions. */
	extern const SO_FGradient SO_kBenchmarkGradients[];

	/* The names of the benchmark functions. */
	extern const char *SO_kBenchmarkName[];

	/* ---------------------------------------------------------------- */

	/* Return a vector with the initialization boundary for a benchmark
	 * problem. The vector is of length n. */
	SO_TElm *SO_BenchmarkLowerInit(size_t problemId, size_t n);
	SO_TElm *SO_BenchmarkUpperInit(size_t problemId, size_t n);

	/* Return a vector with the search-space boundary for a benchmark
	 * problem. The vector is of length n. */
	SO_TElm *SO_BenchmarkLowerBound(size_t problemId, size_t n);
	SO_TElm *SO_BenchmarkUpperBound(size_t problemId, size_t n);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_BENCHMARK_H */
