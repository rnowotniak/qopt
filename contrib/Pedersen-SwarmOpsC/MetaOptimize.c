/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	MetaOptimize
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/MetaOptimize.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>
#include <SwarmOps/Problems/Benchmarks.h>
#include <SwarmOps/Methods/Helpers/Looper.h>
#include <SwarmOps/Methods/Helpers/Printer.h>
#include <SwarmOps/Methods/Methods.h>
#include <stdlib.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

struct SO_MetaSolution SO_MetaOptimize
	(size_t metaMethodId,
	size_t metaNumRuns,
	size_t metaNumIterations,
	void *metaSettings,
	size_t methodId,
	size_t numRuns,
	size_t numIterations,
	void *settings,
	SO_FProblem f,
	SO_FGradient fGradient,
	void *fContext,
	SO_TDim n,
	SO_TElm const* lowerInit,
	SO_TElm const* upperInit,
	SO_TElm const* lowerBound,
	SO_TElm const* upperBound)
{
	/* The solution-struct that is to be returned. */
	struct SO_MetaSolution metaSolution;

	/* Create contexts for optimization and meta-optimization layers. */

	/* Optimization layer. */
	struct SO_MethodContext methodContext = SO_MakeMethodContext(f, fGradient, fContext, n, lowerInit, upperInit, lowerBound, upperBound, numIterations, settings, 0);
	struct SO_LooperContext looperContext = SO_MakeLooperContext(SO_kMethods[methodId], &methodContext, numRuns);

	/* Intermediate printer-layer. */
	struct SO_PrinterContext printerContext = SO_MakePrinterContext(SO_Looper, (void*) &looperContext, SO_kMethodNumParameters[methodId]);

	/* Meta-optimization layer. */
	struct SO_MethodContext metaMethodContext = SO_MakeMethodContext(SO_Printer, 0, (void*) &printerContext, SO_kMethodNumParameters[methodId], SO_kMethodLowerInit[methodId], SO_kMethodUpperInit[methodId], SO_kMethodLowerBound[methodId], SO_kMethodUpperBound[methodId], metaNumIterations, metaSettings, 0);
	struct SO_LooperContext metaLooperContext = SO_MakeLooperContext(SO_kMethods[metaMethodId], &metaMethodContext, metaNumRuns);

	assert(lowerInit);
	assert(upperInit);
	assert(lowerBound);
	assert(upperBound);

	/* Perform actual meta-optimization. */
	SO_Looper(SO_kMethodDefaultParameters[metaMethodId], (void*) &metaLooperContext, SO_kFitnessMax);

	/* Create solution / results to be returned. */
	metaSolution = SO_MakeMetaSolution(&metaMethodContext, &methodContext);

	/* Free contexts. */
	SO_FreeLooperContext(&metaLooperContext);
	SO_FreeMethodContext(&metaMethodContext);
	SO_FreePrinterContext(&printerContext);
	SO_FreeLooperContext(&looperContext);
	SO_FreeMethodContext(&methodContext);

	return metaSolution;
}

/* ---------------------------------------------------------------- */
