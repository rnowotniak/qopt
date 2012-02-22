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
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Statistics/Statistics.h>
#include <SwarmOps/Tools/Vector.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

struct SO_Statistics SO_MakeStatistics(struct SO_LooperLogContext *looperLogContext)
{
	struct SO_Statistics s;
	SO_TFitness *fitnessResults = looperLogContext->fitnessResults;
	size_t numRuns = looperLogContext->numRuns;

	s.fitnessMin = SO_Min(fitnessResults, numRuns);
	s.fitnessMax = SO_Max(fitnessResults, numRuns);
	s.fitnessSum = SO_Sum(fitnessResults, numRuns);
	s.fitnessAvg = SO_Average(fitnessResults, numRuns);
	s.fitnessStdDev = SO_StdDeviation(fitnessResults, numRuns);

	return s;
}

/* ---------------------------------------------------------------- */

struct SO_Statistics SO_CopyStatistics(struct SO_Statistics *s)
{
	/* No allocated memory needs to be copied. */

	return *s;
}

/* ---------------------------------------------------------------- */

void SO_FreeStatistics(struct SO_Statistics *s)
{
	/* Do nothing. */
}

/* ---------------------------------------------------------------- */
