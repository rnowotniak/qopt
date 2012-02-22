/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	HC
 *
 *	Hill-Climber (HC) optimization method originally due to
 *	Metropolis et al. Here made for real-coded search-spaces.
 *	Does local sampling with a stochastic choice on whether
 *	to move, depending on the fitness difference between
 *	current and new potential position.
 *
 * ================================================================ */

#ifndef SO_HC_H
#define SO_HC_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Behavioural parameters and their boundaries. */

#define SO_kNumParametersHC 2

	extern const SO_TElm SO_kParametersDefaultHC[];
	extern const SO_TElm SO_kParametersLowerHC[];
	extern const SO_TElm SO_kParametersUpperHC[];

	/* String containing the name of the optimization method. */
	extern const char SO_kNameHC[];

	/* Array of strings containing the parameter names of the optimization method. */
	extern const char* SO_kParameterNameHC[];

	/* ---------------------------------------------------------------- */

	/* Functions for retrieving the individual parameters from a vector. */

	/* Sampling-range factor. */
	SO_TElm SO_HCRange(const SO_TElm *param);

	/* Movement-probability weight. */
	SO_TElm SO_HCWeight(const SO_TElm *param);

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_HC(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_HC_H */
