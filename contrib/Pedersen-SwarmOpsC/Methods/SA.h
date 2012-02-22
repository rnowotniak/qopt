/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	SA
 *
 *	Simulated Annealing (SA) optimization method originally
 *	due Kirkpatrick et al. Here made for real-coded search-spaces.
 *	Does local sampling with a stochastic choice on whether
 *	to move, depending on the fitness difference between
 *	current and new potential position. The movement probability
 *	is altered during an optimization run, and the agent has its
 *	position in the search-space reset to a random value at
 *	designated intervals.
 *
 * ================================================================ */

#ifndef SO_SA_H
#define SO_SA_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Behavioural parameters and their boundaries. */

#define SO_kNumParametersSA 4

	extern const SO_TElm SO_kParametersDefaultSA[];
	extern const SO_TElm SO_kParametersLowerSA[];
	extern const SO_TElm SO_kParametersUpperSA[];

	/* String containing the name of the optimization method. */
	extern const char SO_kNameSA[];

	/* Array of strings containing the parameter names of the optimization method. */
	extern const char* SO_kParameterNameSA[];

	/* ---------------------------------------------------------------- */

	/* Functions for retrieving the individual parameters from a vector. */

	/* Sampling-range factor. */
	SO_TElm SO_SARange(const SO_TElm *param);

	/* Movement-probability weight start-value. */
	SO_TElm SO_SAAlpha(const SO_TElm *param);

	/* Movement-probability weight end-value. */
	SO_TElm SO_SABeta(const SO_TElm *param);

	/* Iterations between resets. */
	size_t SO_SATime(const SO_TElm *param);

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_SA(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_SA_H */
