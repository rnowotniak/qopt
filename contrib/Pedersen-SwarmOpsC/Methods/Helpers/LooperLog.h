/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	LooperLog
 *
 *	Makes a number of optimization run repeats and sums
 *	the fitness results. Similar to Looper, only this
 *	also stores the results obtained in the individual
 *	optimization runs.
 *
 * ================================================================ */

#ifndef SO_LOOPERLOG_H
#define SO_LOOPERLOG_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* The context-struct holding various data. */
	struct SO_LooperLogContext
	{
		SO_FProblem					f;					/* The method to be repeated. */
		struct SO_MethodContext*	fContext;			/* The method's context. */
		size_t						numRuns;			/* Number of repeats / runs. */
		size_t						numExecuted;		/* Number of runs actually executed. */

		SO_TElm**					results;			/* Results of each run. */
		SO_TFitness*				fitnessResults;		/* Fitness of each run. */
	};

	/* Create and return a looper-context. The arguments are as described
	 * for the struct above. */
	struct SO_LooperLogContext SO_MakeLooperLogContext
		(SO_FProblem f,
		struct SO_MethodContext *fContext,
		size_t numRuns);

	/* Free memory allocated in SO_MakeLooperLogContext. Memory allocated
	 * elsewhere will not be freed, such as fContext. */
	void SO_FreeLooperLogContext(struct SO_LooperLogContext *c);

	/* ---------------------------------------------------------------- */

	/* Performs optimization runs and returns the summed fitness. The
	 * results of individual runs are logged for later analysis. */
	SO_TFitness SO_LooperLog(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_LOOPERLOG_H */
