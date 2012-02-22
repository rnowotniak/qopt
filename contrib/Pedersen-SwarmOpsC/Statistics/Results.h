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
 *	Hold the results and various statistics for a number of
 *	optimization runs.
 *
 * ================================================================ */

#ifndef SO_RESULTS_H
#define SO_RESULTS_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <SwarmOps/Methods/Helpers/LooperLog.h>
#include <SwarmOps/Statistics/Statistics.h>
#include <SwarmOps/Statistics/Solution.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* The struct holding the results and various statistics of a number
	 * of optimization runs. */
	struct SO_Results
	{
		size_t						numRuns;		/* Number of runs. */

		struct SO_Solution			best;			/* Best found solution. */
		struct SO_Statistics		stat;			/* Various statistics. */

		SO_TElm**					results;		/* Results of each run. */
		SO_TFitness*				fitnessResults;	/* Fitnes of each run. */
	};

	/* Create the result-struct by retrieving information from the provided
	 * methodContext and looperLogContext. */
	struct SO_Results SO_MakeResults(struct SO_MethodContext *methodContext, struct SO_LooperLogContext *looperLogContext);

	/* Free the result-struct. */
	void SO_FreeResults(struct SO_Results *r);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_RESULTS_H */
