/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Statistics
 *
 *	Compute and hold various statistics for a number of
 *	optimization runs.
 *
 * ================================================================ */

#ifndef SO_STATISTICS_H
#define SO_STATISTICS_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Methods/Helpers/LooperLog.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* The struct holding the statistics for the optimization runs. */
	struct SO_Statistics
	{
		SO_TFitness fitnessMin;				/* Minimum fitness value for all runs. */
		SO_TFitness fitnessMax;				/* Maximum fitness value for all runs. */
		SO_TFitness fitnessSum;				/* Sum of fitness values for all runs. */
		SO_TFitness fitnessAvg;				/* Average fitness value for the runs. */
		SO_TFitness fitnessStdDev;			/* Std. deviation for the fitness values. */
	};

	/* Compute and return the statistics-struct.
	 * This should be called after executing the optimization runs. */
	struct SO_Statistics SO_MakeStatistics(struct SO_LooperLogContext *looperLogContext);

	/* Return a copy of the given statistics-struct. */
	struct SO_Statistics SO_CopyStatistics(struct SO_Statistics *s);

	/* Free the contents of the statistics-struct. */
	void SO_FreeStatistics(struct SO_Statistics *s);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_STATISTICS_H */
