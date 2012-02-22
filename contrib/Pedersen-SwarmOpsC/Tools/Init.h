/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Init
 *
 *	Functions for initializing vectors with positions
 *	and ranges of the search-space.
 *
 * ================================================================ */

#ifndef SO_INIT_H
#define SO_INIT_H

#include <SwarmOps/Tools/Types.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Initialize vector x having n elements with uniform random values
	 * between the lower and upper boundaries. */
	void SO_InitUniform(SO_TElm *x, SO_TDim n, const SO_TElm *lower, const SO_TElm *upper);

	/* Initialize vector d having n elements with the range between the
	 * lower and upper boundaries. That is, d[i] = upper[i] - lower[i] */
	void SO_InitRange(SO_TElm *d, SO_TDim n, const SO_TElm *lower, const SO_TElm *upper);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_INIT_H */
