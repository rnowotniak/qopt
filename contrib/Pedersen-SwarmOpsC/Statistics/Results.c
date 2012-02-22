/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Results
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Statistics/Results.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Matrix.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

struct SO_Results SO_MakeResults(struct SO_MethodContext *methodContext, struct SO_LooperLogContext *looperLogContext)
{
	struct SO_Results r;

	size_t n = methodContext->fDim;
	size_t numRuns = looperLogContext->numRuns;

	r.numRuns = numRuns;

	r.best = SO_MakeSolution(methodContext);
	r.stat = SO_MakeStatistics(looperLogContext);

	/* Allocate and copy best results from each run. */
	r.results = SO_NewMatrix(numRuns, n);
	SO_CopyMatrix(r.results, looperLogContext->results, numRuns, n);

	/* Allocate and copy fitnesses for the best results from each run. */
	r.fitnessResults = SO_NewFitnessVector(numRuns);
	SO_CopyVector(r.fitnessResults, looperLogContext->fitnessResults, numRuns);

	return r;
}

/* ---------------------------------------------------------------- */

void SO_FreeResults(struct SO_Results *r)
{
	SO_FreeSolution(&r->best);
	SO_FreeStatistics(&r->stat);

	SO_FreeMatrix(r->results, r->numRuns);
	r->results = 0;

	SO_FreeVector(r->fitnessResults);
	r->fitnessResults = 0;
}

/* ---------------------------------------------------------------- */
