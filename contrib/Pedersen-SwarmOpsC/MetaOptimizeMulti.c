/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	MetaOptimizeMulti
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/MetaOptimizeMulti.h>
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

struct SO_Solution SO_MetaOptimizeMulti
	(size_t metaMethodId,
	size_t metaNumRuns,
	size_t metaNumIterations,
	void *metaSettings,
	size_t methodId,
	size_t numRuns,
	size_t *numIterations,
	void *settings,
	size_t numProblems,
	SO_FProblem *f,
	SO_FGradient *fGradient,
	void **contexts,
	SO_TDim *dim,
	SO_TElm **lowerInit,
	SO_TElm **upperInit,
	SO_TElm **lowerBound,
	SO_TElm **upperBound,
	const char *traceFilename)
{
	/* Solution-struct that is to be returned. */
	struct SO_Solution solution;

	/* Create data-structure for holding fitness-trace. */
	struct SO_FitnessTrace fitnessTrace = SO_MakeFitnessTrace(traceFilename, metaNumRuns, metaNumIterations);

	/* Contexts for optimization layer, with support for multiple problems. */
	struct SO_MethodContext *methodContexts = (struct SO_MethodContext*) SO_MAlloc(sizeof(struct SO_MethodContext) * numProblems);
	struct SO_LooperContext *looperContexts = (struct SO_LooperContext*) SO_MAlloc(sizeof(struct SO_LooperContext) * numProblems);
	void **looperContextsPtr = (void**) SO_MAlloc(sizeof(void*) * numProblems);
	struct SO_MultiContext multiContext;

	/* Context for intermediate printer-layer. */
	struct SO_PrinterContext printerContext;

	/* Contexts for meta-optimization layer. */
	struct SO_MethodContext metaMethodContext;
	struct SO_LooperLogContext metaLooperLogContext;

	/* Iteration variable. */
	size_t i;

	/* Initialize contexts for optimization layer. */
	for (i=0; i<numProblems; i++)
	{
		methodContexts[i] = SO_MakeMethodContext(f[i], fGradient[i], contexts[i], dim[i], lowerInit[i], upperInit[i], lowerBound[i], upperBound[i], numIterations[i], settings, 0);
		looperContexts[i] = SO_MakeLooperContext(SO_kMethods[methodId], (void*) &methodContexts[i], numRuns);

		looperContextsPtr[i] = (void*) &looperContexts[i];
	}

	/* Combine contexts for optimization layer into a multi-context. */
	multiContext = SO_MakeMultiContext(SO_Looper, looperContextsPtr, numProblems);

	/* Create context for intermediate printer-layer. */
	printerContext = SO_MakePrinterContext(SO_Multi, (void*) &multiContext, SO_kMethodNumParameters[methodId]);

	/* Create contexts for meta-optimization layer. */
	metaMethodContext = SO_MakeMethodContext(SO_Printer, 0, (void*) &printerContext, SO_kMethodNumParameters[methodId], SO_kMethodLowerInit[methodId], SO_kMethodUpperInit[methodId], SO_kMethodLowerBound[methodId], SO_kMethodUpperBound[methodId], metaNumIterations, metaSettings, &fitnessTrace);
	metaLooperLogContext = SO_MakeLooperLogContext(SO_kMethods[metaMethodId], &metaMethodContext, metaNumRuns);

	/* Perform actual meta-optimization. */
	SO_LooperLog(SO_kMethodDefaultParameters[metaMethodId], (void*) &metaLooperLogContext, SO_kFitnessMax);

	/* Create the solution / results to be returned. */
	solution = SO_MakeSolution(&metaMethodContext);

	/* Write fitness trace to file. */
	SO_WriteFitnessTrace(&fitnessTrace);
	
	/* Free contexts for optimization layer. */
	for (i=0; i<numProblems; i++)
	{
		SO_FreeMethodContext(&methodContexts[i]);
		SO_FreeLooperContext(&looperContexts[i]);
	}

	/* (Continued...) Free contexts for optimization layer. */
	free(methodContexts);
	free(looperContexts);
	free(looperContextsPtr);

	/* Free other contexts. */
	SO_FreeMultiContext(&multiContext);
	SO_FreePrinterContext(&printerContext);
	SO_FreeMethodContext(&metaMethodContext);
	SO_FreeLooperLogContext(&metaLooperLogContext);

	/* Free fitness trace contents. */
	SO_FreeFitnessTrace(&fitnessTrace);

	return solution;
}

/* ---------------------------------------------------------------- */
