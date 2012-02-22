/// ------------------------------------------------------
/// SwarmOps - Numeric and heuristic optimization for C#
/// Copyright (C) 2003-2011 Magnus Erik Hvass Pedersen.
/// Please see the file license.txt for license details.
/// SwarmOps on the internet: http://www.Hvass-Labs.org/
/// ------------------------------------------------------

using System;
using System.Collections.Generic;

using SwarmOps;
using SwarmOps.Problems;
using SwarmOps.Optimizers;

namespace TestMesh
{
    /// <summary>
    /// Compute a mesh of meta-fitness values showing how an
    /// optimizer performs when varying its control parameters.
    /// </summary>
    class Program
    {
        // Mesh settings.
        static readonly int MeshNumIterationsPerDim = 40;

        // Settings for the optimization layer.
        static readonly int NumRuns = 50;
        static readonly int Dim = 5;
        static readonly int DimFactor = 2000;
        static readonly int NumIterations = DimFactor * Dim;

        // Mangle search-space.
        static readonly bool UseMangler = false;
        static readonly double Spillover = 0.05;       // E.g. 0.05
        static readonly double Displacement = 0.1;     // E.g. 0.1
        static readonly double Diffusion = 0.01;       // E.g. 0.01
        static readonly double FitnessNoise = 0.01;    // E.g. 0.01

        // Wrap problem-object in search-space mangler.
        static Problem Mangle(Problem problem)
        {
            return (UseMangler) ? (new Mangler(problem, Diffusion, Displacement, Spillover, FitnessNoise)) : (problem);
        }

        // The optimizer whose control parameters are to be tuned.
        static Optimizer Optimizer = new MOL();
        //static Optimizer Optimizer = new DE();
        //static Optimizer Optimizer = new SwarmOps.Optimizers.Parallel.DESuite(DECrossover.Variant.Rand1Bin, DESuite.DitherVariant.None);

        // Fix certain parameters of the optimizer.
        static double?[] ParameterMask = { 64.0, null, null };

        // Problems to optimize. That is, the optimizer is having its control
        // parameters tuned to work well on these problems.
        static Problem[] Problems =
            new Problem[]
            {
                //Mangle(new Ackley(Dim, NumIterations)),
                Mangle(new Griewank(Dim, NumIterations)),
                Mangle(new Penalized1(Dim, NumIterations)),
                //Mangle(new Penalized2(Dim, NumIterations)),
                //Mangle(new QuarticNoise(Dim, NumIterations)),
                //Mangle(new Rastrigin(Dim, NumIterations)),
                Mangle(new Rosenbrock(Dim, NumIterations)),
                Mangle(new Schwefel12(Dim, NumIterations)),
                //Mangle(new Schwefel221(Dim, NumIterations)),
                //Mangle(new Schwefel222(Dim, NumIterations)),
                //Mangle(new Sphere(Dim, NumIterations)),
                //Mangle(new Step(Dim, NumIterations)),
            };

        // The meta-fitness consists of computing optimization performance
        // for the problems listed above over several optimization runs and
        // sum the results, so we wrap the Optimizer-object in a
        // MetaFitness-object which takes of this.
        static MetaFitness MetaFitness = new MetaFitness(Optimizer, Problems, NumRuns, 0);

        // Print meta-optimization progress.
        static FitnessPrint MetaFitnessPrint = new FitnessPrint(MetaFitness);

        // Log all candidate solutions.
        static int LogCapacity = 20;
        static bool LogOnlyFeasible = false;
        static LogSolutions LogSolutions = new LogSolutions(MetaFitnessPrint, LogCapacity, LogOnlyFeasible);

        // The meta-optimizer.
        static Optimizer MetaOptimizer = new MESH(ParameterMask, LogSolutions);

        // Control parameters to use for the meta-optimizer.
        static double[] MetaParameters = { MeshNumIterationsPerDim };

        // Wrap the meta-optimizer in a Statistics object for logging results.
        static readonly bool StatisticsOnlyFeasible = true;
        static Statistics Statistics = new Statistics(MetaOptimizer, StatisticsOnlyFeasible);

        static void Main(string[] args)
        {
            // Initialize the PRNG.
            //Globals.Random = new RandomOps.MersenneTwister();     // Use with sequential optimizer.
            Globals.Random = new RandomOps.ThreadSafe.CMWC4096();   // Use with parallel optimizer.

            // Output settings.
            Console.WriteLine("Mesh of meta-fitness values using benchmark problems.");
            Console.WriteLine();
            Console.WriteLine("Optimizer to compute mesh for: {0}", Optimizer.Name);
            Console.WriteLine("Mesh, number of iterations per dimension: {0}", MeshNumIterationsPerDim);
            Console.WriteLine("Number of benchmark problems: {0}", Problems.Length);

            for (int i = 0; i < Problems.Length; i++)
            {
                Console.WriteLine("\t{0}", Problems[i].Name);
            }

            Console.WriteLine("Dimensionality for each benchmark problem: {0}", Dim);
            Console.WriteLine("Number of runs per benchmark problem: {0}", NumRuns);
            Console.WriteLine("Number of iterations per run: {0}", NumIterations);
            if (UseMangler)
            {
                Console.WriteLine("Mangle search-space:");
                Console.WriteLine("\tSpillover:     {0}", Spillover);
                Console.WriteLine("\tDisplacement:  {0}", Displacement);
                Console.WriteLine("\tDiffusion:     {0}", Diffusion);
                Console.WriteLine("\tFitnessNoise:  {0}", FitnessNoise);
            }
            else
            {
                Console.WriteLine("Mangle search-space: No");
            }
            Console.WriteLine();

            Console.WriteLine("0/1 Boolean whether optimizer's control parameters are feasible.");
            Console.WriteLine("*** Indicates meta-fitness/feasibility is an improvement.");
            Console.WriteLine();
            Console.WriteLine("Mesh of meta-fitness values:");

            // Start-time.
            DateTime t1 = DateTime.Now;

            // Perform the meta-optimization runs.
            double fitness = Statistics.Fitness(MetaParameters);

            // End-time.
            DateTime t2 = DateTime.Now;

            // Compute result-statistics.
            Statistics.Compute();

            // Retrieve best-found control parameters for the optimizer.
            double[] bestParameters = Statistics.BestResult.Parameters;

            // Output results and statistics.
            Console.WriteLine();
            Console.WriteLine("Best found parameters for {0} optimizer:", Optimizer.Name);
            Tools.PrintParameters(Optimizer, bestParameters);
            Console.WriteLine("Parameters written in array notation:");
            Console.WriteLine("\t{0}", Tools.ArrayToString(bestParameters, 4));
            Console.WriteLine("Best parameters have meta-fitness: {0}", Tools.FormatNumber(Statistics.FitnessMin));

            // Output best found parameters.
            Console.WriteLine();
            Console.WriteLine("Best {0} found parameters:", LogSolutions.Capacity);
            foreach (Solution candidateSolution in LogSolutions.Log)
            {
                Console.WriteLine("\t{0}\t{1}\t{2}",
                    Tools.ArrayToStringRaw(candidateSolution.Parameters, 4),
                    Tools.FormatNumber(candidateSolution.Fitness),
                    (candidateSolution.Feasible) ? (1) : (0));
            }

            // Output time-usage.
            Console.WriteLine();
            Console.WriteLine("Time usage: {0}", t2 - t1);
        }
    }
}
