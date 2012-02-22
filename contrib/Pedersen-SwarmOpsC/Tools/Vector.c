/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Bound
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Tools/Memory.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

SO_TElm *SO_NewVector(size_t size)
{
	return (SO_TElm*) SO_MAlloc(sizeof(SO_TElm)*size);
}

/* ---------------------------------------------------------------- */

SO_TFitness *SO_NewFitnessVector(size_t size)
{
	return (SO_TFitness*) SO_MAlloc(sizeof(SO_TFitness)*size);
}

/* ---------------------------------------------------------------- */

void SO_FreeVector(void *v)
{
	SO_Free(v);
}

/* ---------------------------------------------------------------- */

void SO_PrintVector(const SO_TElm *v, size_t n)
{
	assert(v);

	if (n>0)
	{
		size_t i;

		printf("[%f", v[0]);

		for (i=1; i<n; i++)
		{
			printf(", %f", v[i]);
		}

		printf("]");
	}
}

/* ---------------------------------------------------------------- */

void SO_InitVector(SO_TElm *v, const SO_TElm value, size_t n)
{
	size_t i;

	assert(v);

	for (i=0; i<n; i++)
	{
		v[i] = value;
	}
}

/* ---------------------------------------------------------------- */

void SO_CopyVector(SO_TElm *dest, const SO_TElm *src, size_t n)
{
	size_t i;

	assert(dest);
	assert(src);

	for (i=0; i<n; i++)
	{
		dest[i] = src[i];
	}
}

/* ---------------------------------------------------------------- */

SO_TFitness SO_Min(const SO_TFitness *fitness, size_t n)
{
	size_t i;
	SO_TFitness fitnessMin = SO_kFitnessMax;

	assert(fitness);

	for (i=0; i<n; i++)
	{
		if (fitness[i] < fitnessMin)
		{
			fitnessMin = fitness[i];
		}
	}

	return fitnessMin;
}

/* ---------------------------------------------------------------- */

SO_TFitness SO_Max(const SO_TFitness *fitness, size_t n)
{
	size_t i;
	SO_TFitness fitnessMax = SO_kFitnessMin;

	assert(fitness);

	for (i=0; i<n; i++)
	{
		if (fitness[i] > fitnessMax)
		{
			fitnessMax = fitness[i];
		}
	}

	return fitnessMax;
}

/* ---------------------------------------------------------------- */

SO_TFitness SO_Sum(const SO_TFitness *fitness, size_t n)
{
	size_t i;
	SO_TFitness fitnessSum = 0;

	assert(fitness);

	for (i=0; i<n; i++)
	{
		fitnessSum += fitness[i];
	}

	return fitnessSum;
}

/* ---------------------------------------------------------------- */

SO_TElm SO_Norm(const SO_TElm *v, size_t n)
{
	size_t i;
	SO_TElm sum = 0;

	assert(v);
	assert(n>=0);

	for (i=0; i<n; i++)
	{
		sum += v[i]*v[i];
	}

	return sqrt(sum);
}

/* ---------------------------------------------------------------- */

SO_TElm SO_Distance(const SO_TElm *a, const SO_TElm *b, size_t n)
{
	size_t i;
	SO_TElm sum = 0;

	assert(a && b);
	assert(n>=0);

	for (i=0; i<n; i++)
	{
		SO_TElm d = a[i]-b[i];
		sum += d*d;
	}

	return sqrt(sum);
}

/* ---------------------------------------------------------------- */

SO_TFitness SO_Average(const SO_TFitness *fitness, size_t n)
{
	return SO_Sum(fitness, n)/n;
}

/* ---------------------------------------------------------------- */

SO_TFitness SO_StdDeviation(const SO_TFitness *fitness, size_t n)
{
	size_t i;
	SO_TFitness fitnessAvg = SO_Average(fitness, n);
	SO_TFitness fitnessDevSum = 0;

	assert(fitness);

	for (i=0; i<n; i++)
	{
		SO_TFitness t = fitness[i]-fitnessAvg;
		fitnessDevSum += t*t;
	}

	return sqrt(fitnessDevSum/n);
}

/* ---------------------------------------------------------------- */
