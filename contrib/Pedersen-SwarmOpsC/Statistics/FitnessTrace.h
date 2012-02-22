/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	FitnessTrace
 *
 *	Functions for making a fitness trace showing the
 *	average optimization progress over a number of runs.
 *
 * ================================================================ */

#ifndef SO_FITNESSTRACE_H
#define SO_FITNESSTRACE_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* The struct for a fitness trace. */
	struct SO_FitnessTrace
	{
		char* filename;					/* Output filename for fitness trace. */
		SO_TFitness* trace;				/* Fitness trace for all runs (summed). */
		size_t numRuns;					/* Number of optimization runs. */
		size_t numIterations;			/* Number of iterations per run. */
	};

	/* ---------------------------------------------------------------- */

	/* Make and return a trace-struct.
	 * Parameters correspond to the struct description above. */
	struct SO_FitnessTrace SO_MakeFitnessTrace
		(const char* filename,
		size_t numRuns,
		size_t numIterations);

	/* Free memory allocated by SO_MakeMethodContext(),
	 * but not memory for the struct itself, the boundaries, etc. */
	void SO_FreeFitnessTrace(struct SO_FitnessTrace *trc);

	/*----------------------------------------------------------------*/

	/* Return boolean whether fitness trace is to be made. */
	int SO_UseFitnessTrace(struct SO_FitnessTrace *trc);

	/*----------------------------------------------------------------*/

	/* Set the fitness value for iteration number i. These are
	 * accumulated and will be averaged when the fitness trace is
	 * saved to file. */
	void SO_SetFitnessTrace
		(struct SO_MethodContext *c,
		size_t i,
		SO_TFitness fitness);

	/*----------------------------------------------------------------*/

	/* Write fitness trace to file. */
	void SO_WriteFitnessTrace(struct SO_FitnessTrace *trc);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_FITNESSTRACE_H */

/* ================================================================ */
