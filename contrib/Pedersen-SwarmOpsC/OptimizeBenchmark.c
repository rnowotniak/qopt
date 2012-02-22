/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	OptimizeBenchmark
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/OptimizeBenchmark.h>
#include <SwarmOps/Optimize.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Problems/Benchmarks.h>
#include <SwarmOps/Methods/Methods.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

struct SO_Statistics SO_OptimizeBenchmark	(
	size_t methodId,
	size_t numRuns,
	size_t numIterations,
	void *settings,
	size_t problemId,
	SO_TDim n,
	int displaceOptimum,
	int initFullRange,
	const char *traceFilename)
{
	/* Results and statistics to be returned. */
	struct SO_Results results;
	struct SO_Statistics stat;

	/* Initialization and search-space boundaries for the given benchmark problem. */
	SO_TElm* lowerInit;
	SO_TElm* upperInit;
	SO_TElm* lowerBound = SO_BenchmarkLowerBound(problemId, n);
	SO_TElm* upperBound = SO_BenchmarkUpperBound(problemId, n);

	/* Create context for benchmark problem. */
	struct SO_BenchmarkContext context = SO_MakeBenchmarkContext(n, displaceOptimum);

	/* Initialization boundaries. */
	if (initFullRange)
	{
		lowerInit = lowerBound;
		upperInit = upperBound;
	}
	else
	{
		lowerInit = SO_BenchmarkLowerInit(problemId, n);
		upperInit = SO_BenchmarkUpperInit(problemId, n);
	}

	/* Perform optimization. */
	results = SO_Optimize(methodId, numRuns, numIterations, settings, SO_kBenchmarkProblems[problemId], SO_kBenchmarkGradients[problemId], (void*) &context, n, lowerInit, upperInit, lowerBound, upperBound, traceFilename);

	/* Copy statistics from results of optimization. */
	stat = SO_CopyStatistics(&results.stat);

	/* Free contents of results-struct. */
	SO_FreeResults(&results);

	/* Free boundary vectors. */
	SO_FreeVector(lowerBound);
	SO_FreeVector(upperBound);

	/* Free initialization vectors, if not using full search-space boundaries. */
	if (!initFullRange)
	{
		SO_FreeVector(lowerInit);
		SO_FreeVector(upperInit);
	}

	return results.stat;
}

/* ---------------------------------------------------------------- */
