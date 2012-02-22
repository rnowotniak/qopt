/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Schwefel221
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Problems/Schwefel221.h>
#include <SwarmOps/Tools/Displace.h>
#include <assert.h>
#include <math.h>

/* ---------------------------------------------------------------- */

const char SO_kNameSchwefel221[] = "Schwefel2-21";

/* ---------------------------------------------------------------- */

SO_TElm SO_Schwefel221GetElm(const SO_TElm *x, size_t i, int displaceOptimum)
{
	SO_TElm elm = SO_Displace(x, i, displaceOptimum, SO_kSchwefel221Displace);
	SO_TElm absElm = fabs(elm);

	return absElm;
}

/* ---------------------------------------------------------------- */

SO_TFitness SO_Schwefel221(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit)
{
	struct SO_BenchmarkContext const* c = (struct SO_BenchmarkContext const*) context;
	SO_TDim n = c->n;
	int displaceOptimum = c->displaceOptimum;

	SO_TElm maxValue;
	SO_TDim i;

	assert(x);
	assert(n>=1);

	maxValue = SO_Schwefel221GetElm(x, 0, displaceOptimum);

	for (i=1; i<n; i++)
	{
		SO_TElm elm = SO_Schwefel221GetElm(x, i, displaceOptimum);

		if (elm>maxValue)
		{
			maxValue = elm;
		}
	}

	return maxValue;
}

/* ---------------------------------------------------------------- */

SO_TDim SO_Schwefel221Gradient(const SO_TElm *x, SO_TElm *v, void *context)
{
	/* Gradient not implemented. */
	assert(0);

	return 0;
}

/* ---------------------------------------------------------------- */
