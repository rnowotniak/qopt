/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Methods
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Methods/Methods.h>
#include <SwarmOps/Methods/MESH.h>
#include <SwarmOps/Methods/RND.h>
#include <SwarmOps/Methods/GD.h>
#include <SwarmOps/Methods/GED.h>
#include <SwarmOps/Methods/HC.h>
#include <SwarmOps/Methods/SA.h>
#include <SwarmOps/Methods/PS.h>
#include <SwarmOps/Methods/LUS.h>
#include <SwarmOps/Methods/DE.h>
#include <SwarmOps/Methods/DESuite.h>
#include <SwarmOps/Methods/DETP.h>
#include <SwarmOps/Methods/JDE.h>
#include <SwarmOps/Methods/ELG.h>
#include <SwarmOps/Methods/MYG.h>
#include <SwarmOps/Methods/PSO.h>
#include <SwarmOps/Methods/FAE.h>
#include <SwarmOps/Methods/MOL.h>
#include <SwarmOps/Methods/LICE.h>
#include <stdio.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

const SO_FMethod SO_kMethods[SO_kNumMethods] =	{SO_MESH,
												SO_RND,
												SO_GD,
												SO_GED,
												SO_HC,
												SO_SA,
												SO_PS,
												SO_LUS,
												SO_DE,
												SO_DESuite,
												SO_DETP,
												SO_JDE,
												SO_ELG,
												SO_MYG,
												SO_PSO,
												SO_FAE,
												SO_MOL,
												SO_LICE};

/* ---------------------------------------------------------------- */

const char* SO_kMethodName[SO_kNumMethods] =	{SO_kNameMESH,
												SO_kNameRND,
												SO_kNameGD,
												SO_kNameGED,
												SO_kNameHC,
												SO_kNameSA,
												SO_kNamePS,
												SO_kNameLUS,
												SO_kNameDE,
												SO_kNameDESuite,
												SO_kNameDETP,
												SO_kNameJDE,
												SO_kNameELG,
												SO_kNameMYG,
												SO_kNamePSO,
												SO_kNameFAE,
												SO_kNameMOL,
												SO_kNameLICE};

/* ---------------------------------------------------------------- */

const size_t SO_kMethodNumParameters[SO_kNumMethods] =	{SO_kNumParametersMESH,
														SO_kNumParametersRND,
														SO_kNumParametersGD,
														SO_kNumParametersGED,
														SO_kNumParametersHC,
														SO_kNumParametersSA,
														SO_kNumParametersPS,
														SO_kNumParametersLUS,
														SO_kNumParametersDE,
														SO_kNumParametersDESuite,
														SO_kNumParametersDETP,
														SO_kNumParametersJDE,
														SO_kNumParametersELG,
														SO_kNumParametersMYG,
														SO_kNumParametersPSO,
														SO_kNumParametersFAE,
														SO_kNumParametersMOL,
														SO_kNumParametersLICE};

/* ---------------------------------------------------------------- */

const char** SO_kMethodParameterName[SO_kNumMethods] =	{SO_kParameterNameMESH,
												SO_kParameterNameRND,
												SO_kParameterNameGD,
												SO_kParameterNameGED,
												SO_kParameterNameHC,
												SO_kParameterNameSA,
												SO_kParameterNamePS,
												SO_kParameterNameLUS,
												SO_kParameterNameDE,
												SO_kParameterNameDESuite,
												SO_kParameterNameDETP,
												SO_kParameterNameJDE,
												SO_kParameterNameELG,
												SO_kParameterNameMYG,
												SO_kParameterNamePSO,
												SO_kParameterNameFAE,
												SO_kParameterNameMOL,
												SO_kParameterNameLICE};

/* ---------------------------------------------------------------- */

const SO_TElm* SO_kMethodDefaultParameters[SO_kNumMethods] =	{SO_kParametersDefaultMESH,
																SO_kParametersDefaultRND,
																SO_kParametersDefaultGD,
																SO_kParametersDefaultGED,
																SO_kParametersDefaultHC,
																SO_kParametersDefaultSA,
																SO_kParametersDefaultPS,
																SO_kParametersDefaultLUS,
																SO_kParametersDefaultDE,
																SO_kParametersDefaultDESuite,
																SO_kParametersDefaultDETP,
																SO_kParametersDefaultJDE,
																SO_kParametersDefaultELG,
																SO_kParametersDefaultMYG,
																SO_kParametersDefaultPSO,
																SO_kParametersDefaultFAE,
																SO_kParametersDefaultMOL,
																SO_kParametersDefaultLICE};

/* ---------------------------------------------------------------- */

const SO_TElm* SO_kMethodLowerInit[SO_kNumMethods] = {SO_kParametersLowerMESH,
													SO_kParametersLowerRND,
													SO_kParametersLowerGD,
													SO_kParametersLowerGED,
													SO_kParametersLowerHC,
													SO_kParametersLowerSA,
													SO_kParametersLowerPS,
													SO_kParametersLowerLUS,
													SO_kParametersLowerDE,
													SO_kParametersLowerDESuite,
													SO_kParametersLowerDETP,
													SO_kParametersLowerJDE,
													SO_kParametersLowerELG,
													SO_kParametersLowerMYG,
													SO_kParametersLowerPSO,
													SO_kParametersLowerFAE,
													SO_kParametersLowerMOL,
													SO_kParametersLowerLICE};

/* ---------------------------------------------------------------- */

const SO_TElm* SO_kMethodUpperInit[SO_kNumMethods] = {SO_kParametersUpperMESH,
													SO_kParametersUpperRND,
													SO_kParametersUpperGD,
													SO_kParametersUpperGED,
													SO_kParametersUpperHC,
													SO_kParametersUpperSA,
													SO_kParametersUpperPS,
													SO_kParametersUpperLUS,
													SO_kParametersUpperDE,
													SO_kParametersUpperDESuite,
													SO_kParametersUpperDETP,
													SO_kParametersUpperJDE,
													SO_kParametersUpperELG,
													SO_kParametersUpperMYG,
													SO_kParametersUpperPSO,
													SO_kParametersUpperFAE,
													SO_kParametersUpperMOL,
													SO_kParametersUpperLICE};

/* ---------------------------------------------------------------- */

const SO_TElm* SO_kMethodLowerBound[SO_kNumMethods] =	{SO_kParametersLowerMESH,
														SO_kParametersLowerRND,
														SO_kParametersLowerGD,
														SO_kParametersLowerGED,
														SO_kParametersLowerHC,
														SO_kParametersLowerSA,
														SO_kParametersLowerPS,
														SO_kParametersLowerLUS,
														SO_kParametersLowerDE,
														SO_kParametersLowerDESuite,
														SO_kParametersLowerDETP,
														SO_kParametersLowerJDE,
														SO_kParametersLowerELG,
														SO_kParametersLowerMYG,
														SO_kParametersLowerPSO,
														SO_kParametersLowerFAE,
														SO_kParametersLowerMOL,
														SO_kParametersLowerLICE};

/* ---------------------------------------------------------------- */

const SO_TElm* SO_kMethodUpperBound[SO_kNumMethods] =	{SO_kParametersUpperMESH,
														SO_kParametersUpperRND,
														SO_kParametersUpperGD,
														SO_kParametersUpperGED,
														SO_kParametersUpperHC,
														SO_kParametersUpperSA,
														SO_kParametersUpperPS,
														SO_kParametersUpperLUS,
														SO_kParametersUpperDE,
														SO_kParametersUpperDESuite,
														SO_kParametersUpperDETP,
														SO_kParametersUpperJDE,
														SO_kParametersUpperELG,
														SO_kParametersUpperMYG,
														SO_kParametersUpperPSO,
														SO_kParametersUpperFAE,
														SO_kParametersUpperMOL,
														SO_kParametersUpperLICE};

/* ---------------------------------------------------------------- */

int SO_RequiresGradient(size_t methodId)
{
	return (methodId == SO_kMethodGD) || (methodId == SO_kMethodGED);
}

/* ---------------------------------------------------------------- */

void SO_PrintParameters(const size_t methodId, SO_TElm const* param)
{
	size_t i;
	const size_t kNumParameters = SO_kMethodNumParameters[methodId];

	if (kNumParameters > 0)
	{
		assert(param != 0);

		for (i=0; i<kNumParameters; i++)
		{
			const char** parameterNameArray = SO_kMethodParameterName[methodId];
			const char* parameterName = parameterNameArray[i];
			const SO_TElm p = param[i];

			printf("\t%s = %g\n", parameterName, p);
		}
	}
	else
	{
		printf("\tN/A\n");
	}
}

/* ---------------------------------------------------------------- */
