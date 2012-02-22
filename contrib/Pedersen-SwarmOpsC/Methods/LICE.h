/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	LICE
 *
 *	Layered and Interleaved Co-Evolution (LICE) optimization
 *	method by M.E.H. Pedersen. Consists of two layers of LUS
 *	optimization methods, where the meta-layer is used to adjust
 *	the behavioural parameter of the base-layer in an interleaved
 *	fashion. The parameters for this LICE method are: The initial
 *	parameter (decrease-factor) for the base-layer LUS, the
 *	decrease-factor for the meta-layer, and the number of iterations
 *	to perform of the base-layer for each iteration of the meta-layer.
 *
 *	The LICE method is an experimental method which uses what might
 *	be called meta-adaptation (as opposed to meta-optimization),
 *	because the base-layer LUS is re-using its discoveries between
 *	optimization runs.
 *
 *	The LICE method requires significantly more iterations than
 *	using the LUS method on its own, but may also have greater
 *	adaptability to previously unseen optimization problems, although
 *	this has not yet been documented and may indeed also be a false
 *	notion.
 *
 * ================================================================ */

#ifndef SO_LICE_H
#define SO_LICE_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Behavioural parameters and their boundaries. */

#define SO_kNumParametersLICE 3

	extern const SO_TElm SO_kParametersDefaultLICE[];
	extern const SO_TElm SO_kParametersLowerLICE[];
	extern const SO_TElm SO_kParametersUpperLICE[];

	/* String containing the name of the optimization method. */
	extern const char SO_kNameLICE[];

	/* Array of strings containing the parameter names of the optimization method. */
	extern const char* SO_kParameterNameLICE[];

	/* ---------------------------------------------------------------- */

	/* Functions for retrieving the individual parameters from a vector. */

	/* Decrease-factor for the LUS meta-layer. */
	SO_TElm SO_LICEGamma2(const SO_TElm *param);

	/* Dim-factor for number of base-layer iterations. */
	size_t SO_LICEN(const SO_TElm *param);

	/* Initial decrease-factor for the LUS optimizer. */
	SO_TElm SO_LICEGamma(const SO_TElm *param);

	/* ---------------------------------------------------------------- */

	/* The optimization method. */
	SO_TFitness SO_LICE(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit);

	/* ---------------------------------------------------------------- */

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_LICE_H */

/* ================================================================ */
