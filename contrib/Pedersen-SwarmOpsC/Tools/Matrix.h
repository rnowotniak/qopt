/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Matrix
 *
 *	Functions for allocating and managing matrices.
 *	This is used for having entire swarms (populations) of
 *	agent-vectors.
 *
 * ================================================================ */

#ifndef SO_MATRIX_H
#define SO_MATRIX_H

#include <SwarmOps/Tools/Types.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Allocate and return pointer to a matrix of size numRows * numCols */
	SO_TElm** SO_NewMatrix(size_t numRows, size_t numCols);

	/* Copy the matrix contents of src to dest. Both matrices are assumed
	 * allocated elsewhere and being of size numRows * numCols */
	void SO_CopyMatrix(SO_TElm** dest, SO_TElm** src, size_t numRows, size_t numCols);

	/* Free the matrix pointed to by the mat pointer and having numRows rows. */
	void SO_FreeMatrix(SO_TElm **mat, size_t numRows);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_MATRIX_H */
