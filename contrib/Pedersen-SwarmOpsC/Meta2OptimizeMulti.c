/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Meta2OptimizeMulti
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Meta2OptimizeMulti.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>
#include <SwarmOps/Problems/Benchmarks.h>
#include <SwarmOps/Methods/Helpers/LooperLog.h>
#include <SwarmOps/Methods/Helpers/Looper.h>
#include <SwarmOps/Methods/Helpers/Multi.h>
#include <SwarmOps/Methods/Helpers/Printer.h>
#include <SwarmOps/Methods/Methods.h>
#include <SwarmOps/Tools/Memory.h>
#include <stdlib.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

struct SO_Solution SO_Meta2OptimizeMulti
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
	const size_t *numIterations,
	void **settings,
	size_t numProblems,
	SO_FProblem *f,
	SO_FGradient *fGradient,
	void **contexts,
	SO_TDim *dim,
	SO_TElm **lowerInit,
	SO_TElm **upperInit,
	SO_TElm **lowerBound,
	SO_TElm **upperBound)
{
	/* The solution/results to be returned. */
	struct SO_Solution solution;

	/* Optimizer context matrices allocation,
	 * one for each optimization method and problem combination. */
	struct SO_MethodContext **methodContexts = (struct SO_MethodContext**) SO_MAlloc(sizeof(struct SO_MethodContext*) * numMethods);
	struct SO_LooperContext **looperContexts = (struct SO_LooperContext**) SO_MAlloc(sizeof(struct SO_LooperContext*) * numMethods);
	void ***looperContextsPtr = (void***) SO_MAlloc(sizeof(void**) * numMethods);
	struct SO_MultiContext *multiContexts = (struct SO_MultiContext*) SO_MAlloc(sizeof(struct SO_MultiContext) * numMethods);

	/* Meta-optimizer contexts allocation, one for each optimization method */
	struct SO_MethodContext *metaMethodContexts = (struct SO_MethodContext*) SO_MAlloc(sizeof(struct SO_MethodContext) * numMethods);
	struct SO_LooperContext *metaLooperContexts  = (struct SO_LooperContext*) SO_MAlloc(sizeof(struct SO_LooperContext) * numMethods);
	void **metaLooperContextsPtr = (void**) SO_MAlloc(sizeof(void*) * numMethods);
	struct SO_MultiContext metaMultiContext;

	/* Meta-meta-optimizer contexts. */
	struct SO_PrinterContext printerContext;
	struct SO_MethodContext meta2MethodContext;
	struct SO_LooperLogContext meta2LooperLogContext;

	/* Iteration variables. */
	size_t i, j;

	/* Optimization layer: Allocate and initialize contexts. */
	for (i=0; i<numMethods; i++)
	{
		/* Determine if there are extra settings for the method-context. */
		void *s = (settings != 0) ? (settings[i]) : (0);

		/* Allocate arrays of contexts. */
		methodContexts[i] = (struct SO_MethodContext*) SO_MAlloc(sizeof(struct SO_MethodContext) * numProblems);
		looperContexts[i] = (struct SO_LooperContext*) SO_MAlloc(sizeof(struct SO_LooperContext) * numProblems);
		looperContextsPtr[i] = (void**) SO_MAlloc(sizeof(void*) * numProblems);

		/* Initialize contexts for each method-problem combination. */
		for (j=0; j<numProblems; j++)
		{
			methodContexts[i][j] = SO_MakeMethodContext(f[j], fGradient[j], contexts[j], dim[j], lowerInit[j], upperInit[j], lowerBound[j], upperBound[j], numIterations[j], s, 0);
			looperContexts[i][j] = SO_MakeLooperContext(SO_kMethods[methodId[i]], (void*) &methodContexts[i][j], numRuns);

			looperContextsPtr[i][j] = (void*) &looperContexts[i][j];
		}

		/* Combine contexts for each problem into a multi-context. */
		multiContexts[i] = SO_MakeMultiContext(SO_Looper, looperContextsPtr[i], numProblems);
	}

	/* Meta-optimization layer: Initialize contexts, one for each method in Optimization Layer. */
	for (i=0; i<numMethods; i++)
	{
		metaMethodContexts[i] = SO_MakeMethodContext(SO_Multi, 0, (void*) &multiContexts[i], SO_kMethodNumParameters[methodId[i]], SO_kMethodLowerInit[methodId[i]], SO_kMethodUpperInit[methodId[i]], SO_kMethodLowerBound[methodId[i]], SO_kMethodUpperBound[methodId[i]], metaNumIterations[i], metaSettings, 0);
		metaLooperContexts[i] = SO_MakeLooperContext(SO_kMethods[metaMethodId], &metaMethodContexts[i], metaNumRuns);

		metaLooperContextsPtr[i] = (void*) &metaLooperContexts[i];
	}

	/* Combine contexts for each optimizer into a multi-context. */
	metaMultiContext = SO_MakeMultiContext(SO_Looper, metaLooperContextsPtr, numMethods);

	/* Intermediate printer-layer. */
	printerContext = SO_MakePrinterContext(SO_Multi, (void*) &metaMultiContext, SO_kMethodNumParameters[metaMethodId]);

	/* Meta-meta-optimizer contexts initialization. */
	meta2MethodContext = SO_MakeMethodContext(SO_Printer, 0, (void*) &printerContext, SO_kMethodNumParameters[metaMethodId], SO_kMethodLowerInit[metaMethodId], SO_kMethodUpperInit[metaMethodId], SO_kMethodLowerBound[metaMethodId], SO_kMethodUpperBound[metaMethodId], meta2NumIterations, meta2Settings, 0);
	meta2LooperLogContext = SO_MakeLooperLogContext(SO_kMethods[meta2MethodId], &meta2MethodContext, meta2NumRuns);

	/* Perform actual meta-meta-optimization. */
	SO_LooperLog(SO_kMethodDefaultParameters[meta2MethodId], (void*) &meta2LooperLogContext, SO_kFitnessMax);

	/* Extract the results/solution from meta-meta-optimization runs. */
	solution = SO_MakeSolution(&meta2MethodContext);

	/* Free contexts and arrays for optimizer layer. */
	for (i=0; i<numMethods; i++)
	{
		/* Free contexts. */
		for (j=0; j<numProblems; j++)
		{
			SO_FreeMethodContext(&methodContexts[i][j]);
			SO_FreeLooperContext(&looperContexts[i][j]);
		}

		/* Free arrays. */
		free(methodContexts[i]);
		free(looperContexts[i]);
		free(looperContextsPtr[i]);

		/* Free multi-context. */
		SO_FreeMultiContext(&multiContexts[i]);
	}

	/* Free contexts for meta-optimizer layer. */
	for (i=0; i<numMethods; i++)
	{
		SO_FreeMethodContext(&metaMethodContexts[i]);
		SO_FreeLooperContext(&metaLooperContexts[i]);
	}

	/* Free arrays for optimizer and meta-optimizer layers. */
	free(methodContexts);
	free(looperContexts);
	free(looperContextsPtr);
	free(multiContexts);
	free(metaMethodContexts);
	free(metaLooperContexts);
	free(metaLooperContextsPtr);

	/* Free contexts for meta-, printer-, and meta-meta-layers. */
	SO_FreeMultiContext(&metaMultiContext);
	SO_FreePrinterContext(&printerContext);
	SO_FreeMethodContext(&meta2MethodContext);
	SO_FreeLooperLogContext(&meta2LooperLogContext);

	return solution;
}

/* ---------------------------------------------------------------- */
