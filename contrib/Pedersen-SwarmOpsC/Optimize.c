/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Optimize
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Optimize.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <SwarmOps/Methods/Helpers/LooperLog.h>
#include <SwarmOps/Methods/Methods.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

struct SO_Results SO_Optimize (
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
	SO_TElm const* upperBound,
	const char *traceFilename)
{
	return SO_OptimizePar(
		SO_kMethodDefaultParameters[methodId], 
		methodId,
		numRuns,
		numIterations,
		settings,
		f,
		fGradient,
		fContext,
		n,
		lowerInit,
		upperInit,
		lowerBound,
		upperBound,
		traceFilename);
}

/* ---------------------------------------------------------------- */

struct SO_Results SO_OptimizePar (
	SO_TElm const* par,
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
	SO_TElm const* upperBound,
	const char *traceFilename)
{
	/* Results-struct that is to be returned. */
	struct SO_Results results;

	/* Create data-structure for holding fitness-trace. */
	struct SO_FitnessTrace fitnessTrace = SO_MakeFitnessTrace(traceFilename, numRuns, numIterations);

	/* Create context for optimization method. */
	struct SO_MethodContext methodContext = SO_MakeMethodContext(f, fGradient, fContext, n, lowerInit, upperInit, lowerBound, upperBound, numIterations, settings, &fitnessTrace);
	struct SO_LooperLogContext looperLogContext = SO_MakeLooperLogContext(SO_kMethods[methodId], &methodContext, numRuns);

	/* Perform actual optimization. */
	SO_LooperLog(par, (void*) &looperLogContext, SO_kFitnessMax);

	/* Extract the optimization results. */
	results = SO_MakeResults(&methodContext, &looperLogContext);

	/* Write fitness trace to file. */
	SO_WriteFitnessTrace(&fitnessTrace);
	
	/* Free contents of structs. */
	SO_FreeFitnessTrace(&fitnessTrace);
	SO_FreeMethodContext(&methodContext);
	SO_FreeLooperLogContext(&looperLogContext);

	return results;
}

/* ---------------------------------------------------------------- */
