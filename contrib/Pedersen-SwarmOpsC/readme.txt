SwarmOps - Black-Box Optimization in ANSI C.
Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
Published under the GNU Lesser General Public License.
Please see the file license.txt for license details.
SwarmOps on the internet: http://www.Hvass-Labs.org/


Installation:

The easiest way to install and use SwarmOps with the
Microsoft Visual C++ compiler is to mirror the directory
path of the development computer. All the ANSI C libraries
from Hvass Laboratories are located in the following path
on that computer:

  C:\Users\Magnus\Documents\Development\Libraries\HvassLabs-C

To install the SwarmOps source-code simply un-zip the
archive to this path, and you should be able to open and
use the MS Visual C++ projects directly. You may install
RandomOps the same way.

Please see the SwarmOps manual for other installation
alternatives.


Requirements:

The SwarmOps library requires a Pseudo-Random Number
Generator (PRNG) to work. By default SwarmOps uses the
RandomOps library version 1.2 or newer, which may be
obtained on the internet: http://www.Hvass-Labs.org/
Please consult the SwarmOps manual on how to use another
PRNG.


Compiler Compatibility:

SwarmOps was developed in MS Visual C++ 2005 but should
be compatible with all ANSI C compilers.


Update History:

Version 1.2:
- Small bugfix in Penalized1 benchmark problem.

Version 1.1:
- Added DESuite with most common DE variants by Storn and Price.
- Added DETP method which is DE with Temporal Parameters.
- Added JDE method, a variant of DE due to Brest et al.
- Added MESH method.
- Added Benchmarks2 project.
- Added MeshMetaBenchmarks project.
- Added several benchmark problems.
- Added names for all parameters of optimization methods.
- Added option to initialize in full search-space for benchmark problems.
- Added SO_InitVector() function.
- Added SO_Min() and SO_Max() functions.
- Changed names for DE and MYG parameters.
- Changed ordering of parameters for DE.
- Changed Ackley problem to return fitness zero if negative value occurred.

Version 1.0:
- First release.
