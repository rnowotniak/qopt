/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	PS
 *
 *	Pattern Search (DE) optimization method originally
 *	due to Fermi and Metropolis. A similar idea is due
 *	to Hooke and Jeeves. This variant uses random
 *	selection of which dimension to update and is hence
 *	a slight simplification of the original methods.
 *
 *	Note that the PS method does not have any user-adjustable
 *	parameters.
 *
 * ================================================================ */

#ifndef SO_PS_H
#define SO_PS_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Behavioural parameters and their boundaries. */
	/* (None for the PS method.) */

#define SO_kNumParametersPS 0
#define SO_kParametersDefaultPS 0
#define SO_kParametersLowerPS 0
#define SO_kParametersUpperPS 0

	/* String containing the name of the optimization method. */
	extern const char SO_kNamePS[];

	/* Array of strings containing the parameter names of the optimization method. */
#define SO_kParameterNamePS 0

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_PS(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_PS_H */
