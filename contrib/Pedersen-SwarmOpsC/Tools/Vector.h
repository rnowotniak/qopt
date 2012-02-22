/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Vector
 *
 *	Functions for managing vectors and computing various
 *	mathematical properties of them.
 *
 * ================================================================ */

#ifndef SO_VECTOR_H
#define SO_VECTOR_H

#include <SwarmOps/Tools/Types.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Allocate a vector of the given size. This size is the number of
	 * elements in the vector and NOT the size in bytes. */
	SO_TElm* SO_NewVector(size_t size);

	/* Allocate a vector of fitness values. */
	SO_TFitness* SO_NewFitnessVector(size_t size);

	/* Free a vector. */
	void SO_FreeVector(void* v);

	/* ---------------------------------------------------------------- */

	/* Initialize all elements of vector v to the given value.
	 * Assumes vector v has already been allocated. */
	void SO_InitVector(SO_TElm *v, const SO_TElm value, size_t n);

	/* Copy n elements from src to dest vectors. Assumes the dest
	 * vector has already been allocated. */
	void SO_CopyVector(SO_TElm *dest, const SO_TElm *src, size_t n);

	/* ---------------------------------------------------------------- */

	/* Print the n elements of vector v. */
	void SO_PrintVector(const SO_TElm *v, size_t n);

	/* ---------------------------------------------------------------- */

	/* Return the minium element of a fitness vector with n elements. */
	SO_TFitness SO_Min(const SO_TFitness *fitness, size_t n);

	/* Return the maximum element of a fitness vector with n elements. */
	SO_TFitness SO_Max(const SO_TFitness *fitness, size_t n);

	/* Return the sum of the elements of a fitness vector with n elements. */
	SO_TFitness SO_Sum(const SO_TFitness *fitness, size_t n);

	/* Return the sum of the elements of a fitness vector with n elements. */
	SO_TFitness SO_Sum(const SO_TFitness *fitness, size_t n);

	/* Return the average or mean of the elements of a fitness vector with n elements. */
	SO_TFitness SO_Average(const SO_TFitness *fitness, size_t n);

	/* Return the standard deviation of the elements of a fitness vector with n elements. */
	SO_TFitness SO_StdDeviation(const SO_TFitness *fitness, size_t n);

	/* Return the length or norm of a vector. */
	SO_TElm SO_Norm(const SO_TElm *v, size_t n);

	/* Return the distance between two vectors. */
	SO_TElm SO_Distance(const SO_TElm *a, const SO_TElm *b, size_t n);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_VECTOR_H */
