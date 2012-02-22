/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Looper
 *
 *	Makes a number of optimization run repeats and sums
 *	the fitness results.
 *
 *	Similar to LooperLog, only without result histogram.
 *
 * ================================================================ */

#ifndef SO_LOOPER_H
#define SO_LOOPER_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* The context-struct holding various data. */
	struct SO_LooperContext
	{
		SO_FProblem					f;					/* The method to be repeated. */
		void*						fContext;			/* The method's context. */
		size_t						numRuns;			/* Number of repeats / runs. */
		size_t						numExececuted;		/* Number of runs actually executed. */
	};

	/* Create and return a looper context. Arguments match those in the struct above. */
	struct SO_LooperContext SO_MakeLooperContext
		(SO_FProblem f,
		void *fContext,
		size_t numRuns);

	/* Free memory allocated by SO_MakeLooperContext(). */
	void SO_FreeLooperContext(struct SO_LooperContext *c);

	/* ---------------------------------------------------------------- */

	/* The function to call during optimization. */
	SO_TFitness SO_Looper(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_LOOPER_H */
