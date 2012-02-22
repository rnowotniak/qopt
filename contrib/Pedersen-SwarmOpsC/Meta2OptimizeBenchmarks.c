/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Meta2OptimizeBenchmarks
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Meta2OptimizeBenchmarks.h>
#include <SwarmOps/Meta2OptimizeMulti.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>
#include <SwarmOps/Problems/Benchmarks.h>
#include <SwarmOps/Methods/Methods.h>
#include <SwarmOps/Tools/Memory.h>
#include <stdlib.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

struct SO_Solution SO_Meta2OptimizeBenchmarks
	(size_t meta2MethodId,
	size_t meta2NumRuns,
	size_t meta2NumIterations,
	void *meta2Settings,
	size_t metaMethodId,
	size_t metaNumRuns,
	const size_t *metaNumIterations,
	void *metaSettings,
	size_t numMethods,
	const size_t *methodId,
	size_t numRuns,
	size_t numIterations,
	void **settings,
	const size_t *problemIds,
	size_t numProblems,
	SO_TDim n,
	int displaceOptimum)
{
	/* The solution / results to be returned. */
	struct SO_Solution solution;

	/* Allocate arrays of boundary vectors (actual vectors allocated below). */
	SO_TElm **lowerInit = (SO_TElm**) SO_MAlloc(sizeof(SO_TElm*) * numProblems);
	SO_TElm **upperInit = (SO_TElm**) SO_MAlloc(sizeof(SO_TElm*) * numProblems);
	SO_TElm **lowerBound = (SO_TElm**) SO_MAlloc(sizeof(SO_TElm*) * numProblems);
	SO_TElm **upperBound = (SO_TElm**) SO_MAlloc(sizeof(SO_TElm*) * numProblems);

	/* Allocate arrays for holding problem-info (initialized below). */
	SO_FProblem *f = (SO_FProblem*) SO_MAlloc(sizeof(SO_FProblem*) * numProblems);
	SO_FGradient *fGradient = (SO_FGradient*) SO_MAlloc(sizeof(SO_FGradient*) * numProblems);
	void **contexts = (void**) SO_MAlloc(sizeof(void*) * numProblems);
	SO_TDim *dim = (SO_TDim*) SO_MAlloc(sizeof(SO_TDim) * numProblems);
	size_t *numIter = (size_t*) SO_MAlloc(sizeof(size_t) * numProblems);

	/* Benchmark-problem context. Same for all problems. */
	struct SO_BenchmarkContext benchmarkContext = SO_MakeBenchmarkContext(n, displaceOptimum);

	/* Iteration variable. */
	size_t i;

	/* Perform various allocations for each of the benchmark problems. */
	for (i=0; i<numProblems; i++)
	{
		size_t problemId = problemIds[i];

		/* Allocate and initialize boundary vectors for each benchmark problem. */
		lowerInit[i] = SO_BenchmarkLowerInit(problemId, n);
		upperInit[i] = SO_BenchmarkUpperInit(problemId, n);
		lowerBound[i] = SO_BenchmarkLowerBound(problemId, n);
		upperBound[i] = SO_BenchmarkUpperBound(problemId, n);

		/* Get information for each of the benchmark problems. */
		f[i] = SO_kBenchmarkProblems[problemId];
		fGradient[i] = SO_kBenchmarkGradients[problemId];
		contexts[i] = &benchmarkContext;
		dim[i] = n;

		numIter[i] = numIterations;
	}

	/* Perform actual meta-meta-optimization. */
	solution = SO_Meta2OptimizeMulti
		(meta2MethodId, meta2NumRuns, meta2NumIterations, meta2Settings,
		metaMethodId, metaNumRuns, metaNumIterations, metaSettings,
		numMethods, methodId, numRuns, numIter, settings,
		numProblems,
		f, fGradient, contexts, dim,
		lowerInit, upperInit, lowerBound, upperBound);

	/* Free boundary vectors for benchmark problems. */
	for (i=0; i<numProblems; i++)
	{
		SO_FreeVector(lowerInit[i]);
		SO_FreeVector(upperInit[i]);
		SO_FreeVector(lowerBound[i]);
		SO_FreeVector(upperBound[i]);
	}

	/* Free arrays of pointers. */
	free(lowerInit);
	free(upperInit);
	free(lowerBound);
	free(upperBound);

	/* Free arrays holding information about benchmark problems. */
	free(f);
	free(fGradient);
	free(contexts);
	free(dim);
	free(numIter);

	return solution;
}

/* ---------------------------------------------------------------- */
