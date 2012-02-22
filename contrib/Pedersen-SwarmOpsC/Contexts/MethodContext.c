/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	MethodContext
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Contexts/MethodContext.h>
#include <SwarmOps/Tools/Vector.h>
#include <stdio.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

struct SO_MethodContext SO_MakeMethodContext
	(SO_FProblem f,
	SO_FGradient fGradient,
	void *fContext,
	SO_TDim fDim,
	SO_TElm const* lowerInit,
	SO_TElm const* upperInit,
	SO_TElm const* lowerBound,
	SO_TElm const* upperBound,
	size_t numIterations,
	void *settings,
	struct SO_FitnessTrace *fitnessTrace)
{
	struct SO_MethodContext c;

	c.f = f;
	c.fGradient = fGradient;
	c.fContext = fContext;
	c.fDim = fDim;
	c.lowerInit = lowerInit;
	c.upperInit = upperInit;
	c.lowerBound = lowerBound;
	c.upperBound = upperBound;
	c.numIterations = numIterations;

	c.g = SO_NewVector(fDim);
	c.gFitness = SO_kFitnessMax;

	c.bestPosition = SO_NewVector(fDim);
	c.bestFitness = SO_kFitnessMax;

	c.settings = settings;

	c.fitnessTrace = fitnessTrace;

	return c;
}

/* ---------------------------------------------------------------- */

void SO_FreeMethodContext(struct SO_MethodContext *c)
{
	SO_FreeVector(c->g);
	c->g = 0;

	SO_FreeVector(c->bestPosition);
	c->bestPosition = 0;
}

/* ---------------------------------------------------------------- */

void SO_MethodUpdateBest
	(struct SO_MethodContext *c,
	const SO_TElm *newPosition,
	SO_TFitness newFitness)
{
	if (newFitness < c->bestFitness)
	{
		SO_CopyVector(c->bestPosition, newPosition, c->fDim);
		c->bestFitness = newFitness;
	}
}

/* ---------------------------------------------------------------- */

void SO_MethodSetResult
	(struct SO_MethodContext *c,
	const SO_TElm *position,
	SO_TFitness fitness)
{
	SO_CopyVector(c->g, position, c->fDim);
	c->gFitness = fitness;
}

/* ---------------------------------------------------------------- */
