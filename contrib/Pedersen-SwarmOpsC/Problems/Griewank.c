/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Griewank
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Problems/Griewank.h>
#include <SwarmOps/Tools/Constants.h>
#include <SwarmOps/Tools/Displace.h>
#include <math.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

const char SO_kNameGriewank[] = "Griewank";

/* ---------------------------------------------------------------- */

SO_TFitness SO_Griewank(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit)
{
	struct SO_BenchmarkContext const* c = (struct SO_BenchmarkContext const*) context;
	const SO_TDim n = c->n;
	const int displaceOptimum = c->displaceOptimum;

	SO_TDim i;

	SO_TElm sum=0, prod=1;

	assert(x);
	assert(n>=0);

	for (i=0; i<n; i++)
	{
		SO_TElm elm = SO_Displace(x, i, displaceOptimum, SO_kGriewankDisplace);

		sum += elm*elm;
		prod *= cos(elm/sqrt((SO_TElm)(i+1)));
	}

	return sum/4000 - prod + 1;
}

/* ---------------------------------------------------------------- */

SO_TDim SO_GriewankGradient(const SO_TElm *x, SO_TElm *v, void *context)
{
	struct SO_BenchmarkContext const* c = (struct SO_BenchmarkContext const*) context;
	SO_TDim n = c->n;
	const int displaceOptimum = c->displaceOptimum;

	SO_TDim i, j;

	SO_TElm rec, val2;

	assert(x);
	assert(n>=0);

	for (i=0; i<n; i++)
	{
		SO_TElm elm = SO_Displace(x, i, displaceOptimum, SO_kGriewankDisplace);

		rec = 1.0/sqrt((SO_TElm)(i+1));
		val2 = sin(elm*rec) * rec;

		for (j=0; j<n; j++)
		{
			if (i != j)
			{
				val2 *= cos(x[j]/sqrt((SO_TElm)(j+1)));
			}
		}

		v[i] = elm * 1.0/2000 + val2;
	}

	return n;
}

/* ---------------------------------------------------------------- */
