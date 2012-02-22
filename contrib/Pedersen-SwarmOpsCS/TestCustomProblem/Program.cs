/// ------------------------------------------------------
/// SwarmOps - Numeric and heuristic optimization for C#
/// Copyright (C) 2003-2011 Magnus Erik Hvass Pedersen.
/// Please see the file license.txt for license details.
/// SwarmOps on the internet: http://www.Hvass-Labs.org/
/// ------------------------------------------------------

using System;
using SwarmOps;
using SwarmOps.Optimizers;

namespace TestCustomProblem
{
    /// <summary>
    /// Test an optimizer on a custom problem.
    /// </summary>
    class Program
    {
        // Create an object of the custom problem.
        static Problem Problem = new CustomProblem();

        // Optimization settings.
        static readonly int NumRuns = 50;
        static readonly int DimFactor = 4000;
        static readonly int Dim = Problem.Dimensionality;
        static readonly int NumIterations = DimFactor * Dim;

        // Create optimizer object.
        static Optimizer Optimizer = new DE(Problem);
        //static Optimizer Optimizer = new DESuite(Problem, DECrossover.Variant.Rand1Bin, DESuite.DitherVariant.None);

        // Control parameters for optimizer.
        static readonly double[] Parameters = Optimizer.DefaultParameters;
        //static readonly double[] Parameters = SwarmOps.Optimizers.MOL.Parameters.AllBenchmarks2Dim400IterB;

        // Wrap the optimizer in a logger of result-statistics.
        static readonly bool StatisticsOnlyFeasible = true;
        static Statistics Statistics = new Statistics(Optimizer, StatisticsOnlyFeasible);

        // Wrap it again in a repeater.
        static Repeat Repeat = new RepeatSum(Statistics, NumRuns);

        static void Main(string[] args)
        {
            // Initialize PRNG.
            Globals.Random = new RandomOps.MersenneTwister();

            // Set max number of optimization iterations to perform.
            Problem.MaxIterations = NumIterations;

            // Output optimization settings.
            Console.WriteLine("Optimizer: {0}", Optimizer.Name);
            Console.WriteLine("Using following parameters:");
            Tools.PrintParameters(Optimizer, Parameters);
            Console.WriteLine("Number of optimization runs: {0}", NumRuns);
            Console.WriteLine("Problem: {0}", Problem.Name);
            Console.WriteLine("\tDimensionality: {0}", Dim);
            Console.WriteLine("\tNumIterations per run, max: {0}", NumIterations);
            Console.WriteLine();

            // Create a fitness trace for tracing the progress of optimization.
            int NumMeanIntervals = 3000;
            FitnessTrace fitnessTrace = new FitnessTraceMean(NumIterations, NumMeanIntervals);
            FeasibleTrace feasibleTrace = new FeasibleTrace(NumIterations, NumMeanIntervals, fitnessTrace);

            // Assign the fitness trace to the optimizer.
            Optimizer.FitnessTrace = feasibleTrace;

            // Start-time.
            DateTime t1 = DateTime.Now;

            // Perform optimizations.
            double fitness = Repeat.Fitness(Parameters);

            // End-time.
            DateTime t2 = DateTime.Now;

            if (Statistics.FeasibleFraction > 0)
            {
                // Compute result-statistics.
                Statistics.Compute();

                // Output best result, as well as result-statistics.
                Console.WriteLine("Best feasible solution found:");
                Tools.PrintParameters(Problem, Statistics.BestParameters);
                Console.WriteLine();
                Console.WriteLine("Result Statistics:");
                Console.WriteLine("\tFeasible: \t{0} of solutions found.", Tools.FormatPercent(Statistics.FeasibleFraction));
                Console.WriteLine("\tBest Fitness: \t{0}", Tools.FormatNumber(Statistics.FitnessMin));
                Console.WriteLine("\tWorst: \t\t{0}", Tools.FormatNumber(Statistics.FitnessMax));
                Console.WriteLine("\tMean: \t\t{0}", Tools.FormatNumber(Statistics.FitnessMean));
                Console.WriteLine("\tStd.Dev.: \t{0}", Tools.FormatNumber(Statistics.FitnessStdDev));
                Console.WriteLine();
                Console.WriteLine("Iterations used per run:");
                Console.WriteLine("\tMean: {0}", Tools.FormatNumber(Statistics.IterationsMean));
            }
            else
            {
                Console.WriteLine("No feasible solutions found.");
            }

            // Output time-usage.
            Console.WriteLine();
            Console.WriteLine("Time usage: {0}", t2 - t1);

            // Output fitness and feasible trace.
            string traceFilename = Problem.Name + ".txt";
            fitnessTrace.WriteToFile("FitnessTrace-" + traceFilename);
            feasibleTrace.WriteToFile("FeasibleTrace-" + traceFilename);
        }
    }
}
