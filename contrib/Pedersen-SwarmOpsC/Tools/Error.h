/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Error
 *
 *	Functions for printing error-messages. See source-file (.c)
 *	for actual error-messages.
 *
 * ================================================================ */

#ifndef SO_ERROR_H
#define SO_ERROR_H

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Print error-message an exit with non-zero error number. */
	void SO_Error(const char *msg, int no);

	/* ---------------------------------------------------------------- */

	enum
	{
		SO_kErrNoMemory = 1				/* Out of memory. */
	};

	/*----------------------------------------------------------------*/

	const char SO_kErrMsgMemory[];		/* Out of memory message. */

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_ERROR_H */
