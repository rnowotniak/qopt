/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Printer
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/Helpers/Printer.h>
#include <stdio.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

struct SO_PrinterContext SO_MakePrinterContext(SO_FProblem f, void *fContext, SO_TDim fDim)
{
	struct SO_PrinterContext c;

	c.f = f;
	c.fContext = fContext;
	c.fDim = fDim;

	return c;
}

/* ---------------------------------------------------------------- */

void SO_FreePrinterContext(struct SO_PrinterContext *c)
{
	/* Do nothing. */
}

/* ---------------------------------------------------------------- */

void SO_PrinterDo(const SO_TElm *param,
				  const SO_TDim n,
				  const SO_TFitness fitness,
				  const SO_TFitness fitnessLimit)
{
	SO_TDim i;

	assert(param);
	assert(n>=0);

	for (i=0; i<n; i++)
	{
		printf("%f ", param[i]);
	}

	printf("%g", fitness);

	if (fitness < fitnessLimit)
	{
		printf("\t ***");
	}

	printf("\n");

	/* Flush stdout, this is useful if piping the output and you wish
	 * to study the the output before the entire optimization run is complete. */
	fflush(stdout);
}

/* ---------------------------------------------------------------- */

SO_TFitness SO_Printer(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit)
{
	/* Cast void-ptr context to correct struct-type. */
	struct SO_PrinterContext* c = (struct SO_PrinterContext*) context;

	/* Clone context to local variables for easier reference. */
	SO_FProblem f = c->f;
	void *fContext = c->fContext;
	SO_TDim fDim = c->fDim;

	SO_TFitness fitness = f(param, fContext, fitnessLimit);

	SO_PrinterDo(param, fDim, fitness, fitnessLimit);

	return fitness;
}

/* ---------------------------------------------------------------- */
