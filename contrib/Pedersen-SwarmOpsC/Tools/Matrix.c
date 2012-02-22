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
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Tools/Vector.h>
#include <SwarmOps/Tools/Matrix.h>
#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Tools/Memory.h>
#include <stdlib.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

SO_TElm** SO_NewMatrix(size_t numRows, size_t numCols)
{
	size_t i;

	/* Allocate an array of pointers. */
	SO_TElm **mat = (SO_TElm**) SO_MAlloc(sizeof(SO_TElm*)*numRows);

	/* Ensure all sub-array pointers are zero,
	 * in case of exception in the allocation of one of them. */
	for (i=0; i<numRows; i++)
	{
		mat[i] = 0;
	}

	/* Now allocate row-vectors. */
	for (i=0; i<numRows; i++)
	{
		mat[i] = SO_NewVector(numCols);

		/* NewVector() should not return if allocation fails.
		 * This leaves the mat-array dangling! Usually this
		 * will not be a problem if the program just exits. */

		assert(mat[i]);
	}

	return mat;
}

/* ---------------------------------------------------------------- */

void SO_FreeMatrix(SO_TElm **mat, size_t numRows)
{
	size_t i;

	assert(mat);

	for (i=0; i<numRows; i++)
	{
		SO_FreeVector(mat[i]);
	}

	free(mat);
}

/* ---------------------------------------------------------------- */

void SO_CopyMatrix(SO_TElm** dest, SO_TElm** src, size_t numRows, size_t numCols)
{
	size_t i;

	assert(dest);
	assert(src);

	for (i=0; i<numRows; i++)
	{
		SO_CopyVector(dest[i], src[i], numCols);
	}
}

/* ---------------------------------------------------------------- */
