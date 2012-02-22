/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Benchmarks
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Problems/Benchmarks.h>
#include <SwarmOps/Problems/Sphere.h>
#include <SwarmOps/Problems/Schwefel222.h>
#include <SwarmOps/Problems/Schwefel12.h>
#include <SwarmOps/Problems/Schwefel221.h>
#include <SwarmOps/Problems/Rosenbrock.h>
#include <SwarmOps/Problems/Step.h>
#include <SwarmOps/Problems/QuarticNoise.h>
#include <SwarmOps/Problems/Rastrigin.h>
#include <SwarmOps/Problems/Ackley.h>
#include <SwarmOps/Problems/Griewank.h>
#include <SwarmOps/Problems/Penalized1.h>
#include <SwarmOps/Problems/Penalized2.h>
#include <SwarmOps/Tools/Vector.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

const SO_FProblem SO_kBenchmarkProblems[SO_kNumBenchmark] =
									{SO_Sphere,
									SO_Schwefel222,
									SO_Schwefel12,
									SO_Schwefel221,
									SO_Griewank,
									SO_Penalized1,
									SO_Penalized2,
									SO_Rastrigin,
									SO_Ackley,
									SO_Rosenbrock,
									SO_Step,
									SO_QuarticNoise};

/* ---------------------------------------------------------------- */

const SO_FGradient SO_kBenchmarkGradients[SO_kNumBenchmark] =
									{SO_SphereGradient,
									SO_Schwefel222Gradient,
									SO_Schwefel12Gradient,
									SO_Schwefel221Gradient,
									SO_GriewankGradient,
									SO_Penalized1Gradient,
									SO_Penalized2Gradient,
									SO_RastriginGradient,
									SO_AckleyGradient,
									SO_RosenbrockGradient,
									SO_StepGradient,
									SO_QuarticNoiseGradient};

/* ---------------------------------------------------------------- */

const char *SO_kBenchmarkName[SO_kNumBenchmark] =
									{SO_kNameSphere,
									SO_kNameSchwefel222,
									SO_kNameSchwefel12,
									SO_kNameSchwefel221,
									SO_kNameGriewank,
									SO_kNamePenalized1,
									SO_kNamePenalized2,
									SO_kNameRastrigin,
									SO_kNameAckley,
									SO_kNameRosenbrock,
									SO_kNameStep,
									SO_kNameQuarticNoise};

/* ---------------------------------------------------------------- */

const SO_TElm SO_kBenchmarkLowerInit[SO_kNumBenchmark] =
									{SO_kSphereLowerInit,
									SO_kSchwefel222LowerInit,
									SO_kSchwefel12LowerInit,
									SO_kSchwefel221LowerInit,
									SO_kGriewankLowerInit,
									SO_kPenalized1LowerInit,
									SO_kPenalized2LowerInit,
									SO_kRastriginLowerInit,
									SO_kAckleyLowerInit,
									SO_kRosenbrockLowerInit,
									SO_kStepLowerInit,
									SO_kQuarticNoiseLowerInit};

/* ---------------------------------------------------------------- */

const SO_TElm SO_kBenchmarkUpperInit[SO_kNumBenchmark] =
									{SO_kSphereUpperInit,
									SO_kSchwefel222UpperInit,
									SO_kSchwefel12UpperInit,
									SO_kSchwefel221UpperInit,
									SO_kGriewankUpperInit,
									SO_kPenalized1UpperInit,
									SO_kPenalized2UpperInit,
									SO_kRastriginUpperInit,
									SO_kAckleyUpperInit,
									SO_kRosenbrockUpperInit,
									SO_kStepUpperInit,
									SO_kQuarticNoiseUpperInit};

/* ---------------------------------------------------------------- */

const SO_TElm SO_kBenchmarkLowerBound[SO_kNumBenchmark] =
									{SO_kSphereLowerBound,
									SO_kSchwefel222LowerBound,
									SO_kSchwefel12LowerBound,
									SO_kSchwefel221LowerBound,
									SO_kGriewankLowerBound,
									SO_kPenalized1LowerBound,
									SO_kPenalized2LowerBound,
									SO_kRastriginLowerBound,
									SO_kAckleyLowerBound,
									SO_kRosenbrockLowerBound,
									SO_kStepLowerBound,
									SO_kQuarticNoiseLowerBound};

/* ---------------------------------------------------------------- */

const SO_TElm SO_kBenchmarkUpperBound[SO_kNumBenchmark] =
									{SO_kSphereUpperBound,
									SO_kSchwefel222UpperBound,
									SO_kSchwefel12UpperBound,
									SO_kSchwefel221UpperBound,
									SO_kGriewankUpperBound,
									SO_kPenalized1UpperBound,
									SO_kPenalized2UpperBound,
									SO_kRastriginUpperBound,
									SO_kAckleyUpperBound,
									SO_kRosenbrockUpperBound,
									SO_kStepUpperBound,
									SO_kQuarticNoiseUpperBound};

/* ---------------------------------------------------------------- */

/* Helper function for allocating and initializing a vector. */

SO_TElm *SO_NewBenchmarkVector(SO_TElm value, size_t n)
{
	SO_TElm* vec = SO_NewVector(n);
	size_t i;

	for (i=0; i<n; i++)
	{
		vec[i] = value;
	}

	return vec;
}

/* ---------------------------------------------------------------- */

SO_TElm *SO_BenchmarkLowerInit(size_t problemId, size_t n)
{
	assert(problemId >= 0 && problemId < SO_kNumBenchmark);

	return SO_NewBenchmarkVector(SO_kBenchmarkLowerInit[problemId], n);
}

/* ---------------------------------------------------------------- */

SO_TElm *SO_BenchmarkUpperInit(size_t problemId, size_t n)
{
	assert(problemId >= 0 && problemId < SO_kNumBenchmark);

	return SO_NewBenchmarkVector(SO_kBenchmarkUpperInit[problemId], n);
}

/* ---------------------------------------------------------------- */

SO_TElm *SO_BenchmarkLowerBound(size_t problemId, size_t n)
{
	assert(problemId >= 0 && problemId < SO_kNumBenchmark);

	return SO_NewBenchmarkVector(SO_kBenchmarkLowerBound[problemId], n);
}

/* ---------------------------------------------------------------- */

SO_TElm *SO_BenchmarkUpperBound(size_t problemId, size_t n)
{
	assert(problemId >= 0 && problemId < SO_kNumBenchmark);

	return SO_NewBenchmarkVector(SO_kBenchmarkUpperBound[problemId], n);
}

/* ---------------------------------------------------------------- */
