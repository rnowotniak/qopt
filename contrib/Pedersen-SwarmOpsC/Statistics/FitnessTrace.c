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
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Statistics/FitnessTrace.h>
#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/String.h>
#include <stdio.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

struct SO_FitnessTrace SO_MakeFitnessTrace
	(const char* filename,
	size_t numRuns,
	size_t numIterations)
{
	struct SO_FitnessTrace trc;
	size_t i;

	trc.filename = SO_CopyString(filename);
	trc.numRuns = numRuns;
	trc.numIterations = numIterations;

	if (SO_UseFitnessTrace(&trc))
	{
		/* Allocate fitness trace. */
		trc.trace = SO_NewFitnessVector(numIterations);

		/* Initialize all elements of the fitness trace to zero. */
		for (i=0; i<numIterations; i++)
		{
			trc.trace[i] = 0;
		}
	}
	else
	{
		trc.trace = 0;
	}

	return trc;
}

/* ---------------------------------------------------------------- */

void SO_FreeFitnessTrace(struct SO_FitnessTrace *trc)
{
	assert(trc);

	SO_FreeString(trc->filename);
	trc->filename = 0;

	/* Delete fitness trace array. */
	SO_FreeVector(trc->trace);
	trc->trace = 0;
}

/* ---------------------------------------------------------------- */

int SO_UseFitnessTrace(struct SO_FitnessTrace *trc)
{
	/* First check if trc-pointer is valid at all,
	 * then check if filename is valid.
	 * Recall that ANSI C has lazy evaluation of boolean
	 * expressions, so if the first expression fails,
	 * then the second is not evaluated. */
	return (trc != 0 && trc->filename != 0);
}

/* ---------------------------------------------------------------- */

void SO_WriteFitnessTrace(struct SO_FitnessTrace *trc)
{
	if (SO_UseFitnessTrace(trc))
	{
		size_t i;
		errno_t err;
		FILE *stream;

		/* Create file. */
		err = fopen_s(&stream, trc->filename, "w");

		if (err == 0)
		{
			/* Write trace to file. */
			for (i=0; i<trc->numIterations; i++)
			{
				fprintf(stream, "%d %e\n", i+1, trc->trace[i] / trc->numRuns);
			}

			/* Close file. */
			fclose(stream);
		}
	}
}

/* ---------------------------------------------------------------- */

void SO_SetFitnessTrace
	(struct SO_MethodContext *c,
	size_t i,
	SO_TFitness fitness)
{
	struct SO_FitnessTrace *trc;

	assert(c);

	trc = c->fitnessTrace;

	if (SO_UseFitnessTrace(trc))
	{
		assert(i < trc->numIterations);

		trc->trace[i] += fitness;
	}
}

/* ---------------------------------------------------------------- */
