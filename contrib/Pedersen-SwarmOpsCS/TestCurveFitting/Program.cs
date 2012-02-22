/// ------------------------------------------------------
/// SwarmOps - Numeric and heuristic optimization for C#
/// Copyright (C) 2003-2011 Magnus Erik Hvass Pedersen.
/// Please see the file license.txt for license details.
/// SwarmOps on the internet: http://www.Hvass-Labs.org/
/// ------------------------------------------------------

using System;
using System.Linq;
using SwarmOps;
using SwarmOps.Optimizers;
using SwarmOps.Problems;

namespace TestCurveFitting
{
    /// <summary>
    /// Test an optimizer for use in curve-fitting.
    /// </summary>
    class Program
    {
        // Raw X- and Y-axis values.
        static readonly double[] X = new double[] { 0, -1, -2, -3, -4 };
        static readonly double[] Y2 = new double[] { 1024, 604, 355, 209, 123 };

        // Normalize Y-axis values.
        static readonly double minY = Y2.Min();
        static readonly double[] Y = Y2.Select(o => o / minY).ToArray();
        
        // Search-space boundaries for curve-fitting parameters.
        static readonly double MinA = -10;
        static readonly double MaxA = 10;
        static readonly double MinB = 0;
        static readonly double MaxB = Y.Max()*2;

        // Create object for curve-fitting problem.
        static CurveFitting Problem = new CurveFittingLin(X, Y, MinA, MaxA, MinB, MaxB);
        //static CurveFitting Problem = new CurveFittingExp(X, Y, MinA, MaxA, MinB, MaxB);

        // Optimization settings.
        static readonly int NumRuns = 1000;
        static readonly int Dim = Problem.Dimensionality;
        static readonly int DimFactor = 50;
        static readonly int NumIterations = DimFactor * Dim;

        // Create optimizer.
        static Optimizer Optimizer = new PS(Problem);

        static void Main(string[] args)
        {
            // Create and initialize PRNG.
            Globals.Random = new RandomOps.MersenneTwister();

            // Set the max number of optimization iterations to perform.
            Problem.MaxIterations = NumIterations;

            // Wrap the optimizer in a logger of result-statistics.
            bool StatisticsOnlyFeasible = true;
            Statistics Statistics = new Statistics(Optimizer, StatisticsOnlyFeasible);

            // Wrap it again in a repeater.
            Repeat repeat = new RepeatSum(Statistics, NumRuns);

            // Output optimization settings.
            Console.WriteLine("Problem: {0}", Problem.Name);
            Console.WriteLine("Optimizer: {0}", Optimizer.Name);
            Console.WriteLine("Using following parameters:");
            Tools.PrintParameters(Optimizer, Optimizer.DefaultParameters);
            Console.WriteLine("Number of runs: {0}", NumRuns);
            Console.WriteLine("Dimensionality: {0}", Dim);
            Console.WriteLine("Dim-factor: {0}", DimFactor);
            Console.WriteLine();

            // Start-time.
            DateTime t1 = DateTime.Now;

            // Perform optimizations.
            double fitness = repeat.Fitness();

            // End-time.
            DateTime t2 = DateTime.Now;

            // Compute result-statistics.
            Statistics.Compute();

            // Output best results, as well as result-statistics.
            Console.WriteLine("Best parameters found:");
            Tools.PrintParameters(Problem, Statistics.BestParameters);
            Console.WriteLine("With fitness: {0}", Tools.FormatNumber(Statistics.FitnessMin));
            Console.WriteLine("Time usage: {0}", t2 - t1);
            Console.WriteLine("Mean number of iterations: {0}", Statistics.IterationsMean);
            Console.WriteLine();
            Console.WriteLine("x\ty\tComputedY");
            Console.WriteLine("--------------------");

            // Output fitted curve.
            for (int i=0; i<X.Length; i++)
            {
                double x = X[i];
                double y = Y[i];
                double computedY = Problem.ComputeY(Statistics.BestParameters, x);

                Console.WriteLine("{0}\t{1}\t{2}",
                    Tools.FormatNumber(x),
                    Tools.FormatNumber(y),
                    Tools.FormatNumber(computedY));
            }
        }
    }
}
