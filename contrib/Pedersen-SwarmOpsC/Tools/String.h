/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	String
 *
 *	Functions for managing textual strings.
 *
 * ================================================================ */

#ifndef SO_STRING_H
#define SO_STRING_H

#include <SwarmOps/Tools/Types.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Allocate and return a copy of the string s. */
	char *SO_CopyString(const char *s);

	/* Free a string. */
	void SO_FreeString(char *s);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_STRING_H */
