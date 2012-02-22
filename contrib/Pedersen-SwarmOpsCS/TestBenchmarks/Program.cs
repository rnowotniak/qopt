/// ------------------------------------------------------
/// SwarmOps - Numeric and heuristic optimization for C#
/// Copyright (C) 2003-2011 Magnus Erik Hvass Pedersen.
/// Please see the file license.txt for license details.
/// SwarmOps on the internet: http://www.Hvass-Labs.org/
/// ------------------------------------------------------

using System;
using SwarmOps;
using SwarmOps.Optimizers;
using SwarmOps.Problems;

namespace TestBenchmarks
{
    /// <summary>
    /// Test an optimizer on various benchmark problems.
    /// </summary>
    class Program
    {
        // Create optimizer object.
        static Optimizer Optimizer = new DE();

        // Control parameters for optimizer.
        static readonly double[] Parameters = Optimizer.DefaultParameters;
        //static readonly double[] Parameters = PSO.Parameters.AllBenchmarks5Dim10000IterA;

        // Optimization settings.
        static readonly int NumRuns = 50;
        static readonly int Dim = 5;
        static readonly int DimFactor = 2000;
        static readonly int NumIterations = DimFactor* Dim;

        // Mangle search-space.
        static readonly bool UseMangler = false;
        static readonly double Spillover = 0.05;       // E.g. 0.05
        static readonly double Displacement = 0.1;     // E.g. 0.1
        static readonly double Diffusion = 0.01;       // E.g. 0.01
        static readonly double FitnessNoise = 0.01;    // E.g. 0.01

        /// <summary>
        /// Optimize the given problem and output result-statistics.
        /// </summary>
        static void Optimize(Problem problem)
        {
            if (UseMangler)
            {
                // Wrap problem-object in search-space mangler.
                problem = new Mangler(problem, Diffusion, Displacement, Spillover, FitnessNoise);
            }

            // Create a fitness trace for tracing the progress of optimization re. mean.
            int NumMeanIntervals = 3000;
            FitnessTrace fitnessTraceMean = new FitnessTraceMean(NumIterations, NumMeanIntervals);

            // Create a fitness trace for tracing the progress of optimization re. quartiles.
            // Note that fitnessTraceMean is chained to this object by passing it to the
            // constructor, this causes both fitness traces to be used.
            int NumQuartileIntervals = 10;
            FitnessTrace fitnessTraceQuartiles = new FitnessTraceQuartiles(NumRuns, NumIterations, NumQuartileIntervals, fitnessTraceMean);

            // Create a feasibility trace for tracing the progress of optimization re. fesibility.
            FeasibleTrace feasibleTrace = new FeasibleTrace(NumIterations, NumMeanIntervals, fitnessTraceQuartiles);

            // Assign the problem etc. to the optimizer.
            Optimizer.Problem = problem;
            Optimizer.FitnessTrace = feasibleTrace;

            // Wrap the optimizer in a logger of result-statistics.
            bool StatisticsOnlyFeasible = true;
            Statistics Statistics = new Statistics(Optimizer, StatisticsOnlyFeasible);

            // Wrap it again in a repeater.
            Repeat Repeat = new RepeatSum(Statistics, NumRuns);

            // Perform the optimization runs.
            double fitness = Repeat.Fitness(Parameters);

            // Compute result-statistics.
            Statistics.Compute();

            // Output result-statistics.
            Console.WriteLine("{0} & {1} & {2} & {3} & {4} & {5} & {6} & {7} & {8} \\\\",
                problem.Name,
                Tools.FormatNumber(Statistics.FitnessMean),
                Tools.FormatNumber(Statistics.FitnessStdDev),
                Tools.FormatNumber(Statistics.FitnessQuartiles.Min),
                Tools.FormatNumber(Statistics.FitnessQuartiles.Q1),
                Tools.FormatNumber(Statistics.FitnessQuartiles.Median),
                Tools.FormatNumber(Statistics.FitnessQuartiles.Q3),
                Tools.FormatNumber(Statistics.FitnessQuartiles.Max),
                Tools.FormatPercent(Statistics.FeasibleFraction));

            // Output fitness and feasible traces.
            fitnessTraceMean.WriteToFile(Optimizer.Name + "-FitnessTraceMean-" + problem.Name + ".txt");
            fitnessTraceQuartiles.WriteToFile(Optimizer.Name + "-FitnessTraceQuartiles-" + problem.Name + ".txt");
            feasibleTrace.WriteToFile(Optimizer.Name + "-FeasibleTrace-" + problem.Name + ".txt");
        }

        static void Main(string[] args)
        {
            Int32 a = -123456789, b = 234567890;
            UInt32 aU = (UInt32)a, bU = (UInt32)b;

            Int32 c = a + b;
            UInt32 cU = aU + bU;

            // Initialize PRNG.
            Globals.Random = new RandomOps.MersenneTwister();

            // Output optimization settings.
            Console.WriteLine("Benchmark-tests.");
            Console.WriteLine("Optimizer: {0}", Optimizer.Name);
            Console.WriteLine("Using following parameters:");
            Tools.PrintParameters(Optimizer, Parameters);
            Console.WriteLine("Number of runs per problem: {0}", NumRuns);
            Console.WriteLine("Dimensionality: {0}", Dim);
            Console.WriteLine("Dim-factor: {0}", DimFactor);
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
            Console.WriteLine("Problem & Mean & Std.Dev. & Min & Q1 & Median & Q3 & Max & Feasible \\\\");
            Console.WriteLine("\\hline");

            // Starting-time.
            DateTime t1 = DateTime.Now;

#if false
            //Optimize(new Ackley(Dim, NumIterations));
            //Optimize(new Rastrigin(Dim, NumIterations));
            //Optimize(new Griewank(Dim, NumIterations));
            //Optimize(new Rosenbrock(Dim, NumIterations));
            //Optimize(new Schwefel12(Dim, NumIterations));
            Optimize(new Sphere(Dim, NumIterations));
            //Optimize(new Step(Dim, NumIterations));
#else
            // Optimize all benchmark problems.
            foreach (Benchmarks.ID benchmarkID in Benchmarks.IDs)
            {
                // Create a new instance of the benchmark problem.
                Problem problem = benchmarkID.CreateInstance(Dim, NumIterations);

                // Optimize the problem.
                Optimize(problem);
            }
#endif
            // End-time.
            DateTime t2 = DateTime.Now;

            // Output time-usage.
            Console.WriteLine();
            Console.WriteLine("Time usage: {0}", t2 - t1);
        }
    }
}
