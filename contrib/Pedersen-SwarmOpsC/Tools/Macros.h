/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Macros
 *
 *	Macros such as finding the max and min of two values.
 *
 * ================================================================ */

#ifndef SO_MACROS_H
#define SO_MACROS_H

#ifdef  __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------- */

#define SO_Max(a, b) ((a)>(b) ? (a) : (b))
#define SO_Min(a, b) ((a)<(b) ? (a) : (b))

/* ---------------------------------------------------------------- */

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_MACROS_H */
