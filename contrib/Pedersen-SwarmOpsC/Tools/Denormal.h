/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Denormal
 *
 *	Functions for flushing denormalized floating point numbers
 *	to zero. Denormalized numbers cause great waste of time
 *	on some CPU's (e.g. Intel Pentium).
 *
 * ================================================================ */

#ifndef SO_DENORMAL_H
#define SO_DENORMAL_H

#include <SwarmOps/Tools/Types.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* If the number x[i] is denormalized flush it to zero. */
	void SO_DenormalFixOne(SO_TElm *x, SO_TDim i);

	/* Same as SO_DenormalFixOne() for all elements of vector x. */
	void SO_DenormalFixAll(SO_TElm *x, SO_TDim n);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_DENORMAL_H */
