/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	QuarticNoise
 *
 *	The QuarticNoise benchmark optimization problem.
 *
 * ================================================================ */

#ifndef SO_QUARTICNOISE_H
#define SO_QUARTICNOISE_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/BenchmarkContext.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Initialization and search-space boundaries. */
#define SO_kQuarticNoiseLowerInit 0.64
#define SO_kQuarticNoiseUpperInit 1.28
#define SO_kQuarticNoiseLowerBound -1.28
#define SO_kQuarticNoiseUpperBound 1.28
#define SO_kQuarticNoiseDisplace -0.32

	/*----------------------------------------------------------------*/

	/* A string holding the problem's name. */
	extern const char SO_kNameQuarticNoise[];

	/* ---------------------------------------------------------------- */

	/* The fitness function. */
	SO_TFitness SO_QuarticNoise(const SO_TElm *x, void *context, const SO_TFitness fitnessLimit);

	/* The gradient of the fitness function. */
	SO_TDim SO_QuarticNoiseGradient(const SO_TElm *x, SO_TElm *v, void *context);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_QUARTICNOISE_H */
