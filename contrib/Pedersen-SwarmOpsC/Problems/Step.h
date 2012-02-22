/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Step
 *
 *	The Step benchmark optimization problem.
 *
 * ================================================================ */

#ifndef SO_STEP_H
#define SO_STEP_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Initialization and search-space boundaries. */
#define SO_kStepLowerInit 50
#define SO_kStepUpperInit 100
#define SO_kStepLowerBound -100
#define SO_kStepUpperBound 100
#define SO_kStepDisplace 25

	/*----------------------------------------------------------------*/

	/* A string holding the problem's name. */
	extern const char SO_kNameStep[];

	/* ---------------------------------------------------------------- */

	/* The fitness function. */
	SO_TFitness SO_Step(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit);

	/* The gradient of the fitness function. */
	SO_TDim SO_StepGradient(const SO_TElm *x, SO_TElm *v, void *context);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_STEP_H */
