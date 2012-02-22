/// ------------------------------------------------------
/// SwarmOps - Numeric and heuristic optimization for C#
/// Copyright (C) 2003-2011 Magnus Erik Hvass Pedersen.
/// Please see the file license.txt for license details.
/// SwarmOps on the internet: http://www.Hvass-Labs.org/
/// ------------------------------------------------------

using System;
using System.Diagnostics;

namespace TestCustomProblem
{
    /// <summary>
    /// Constrained optimization problem, example.
    /// This is the 2-dimensional Rosenbrock problem
    /// with some example constraints.
    /// The optimal feasible solution seems to be:
    /// a ~ 1.5937
    /// b ~ 2.5416
    /// </summary>
    class CustomProblem : SwarmOps.Problem
    {
        #region Constructors.
        /// <summary>
        /// Construct the object.
        /// </summary>
        public CustomProblem()
            : base()
        {
        }
        #endregion

        #region Get parameters.
        /// <summary>
        /// Get parameter, A.
        /// </summary>
        /// <param name="parameters">Optimizer parameters.</param>
        public double GetA(double[] parameters)
        {
            return parameters[0];
        }

        /// <summary>
        /// Get parameter, B.
        /// </summary>
        /// <param name="parameters">Optimizer parameters.</param>
        public double GetB(double[] parameters)
        {
            return parameters[1];
        }
        #endregion

        #region Base-class overrides.
        /// <summary>
        /// Name of the optimizer.
        /// </summary>
        public override string Name
        {
            get { return "CustomProblem"; }
        }

        /// <summary>
        /// Dimensionality of the problem.
        /// </summary>
        public override int Dimensionality
        {
            get { return 2; }
        }

        double[] _lowerBound = { -100, -100 };

        /// <summary>
        /// Lower search-space boundary.
        /// </summary>
        public override double[] LowerBound
        {
            get { return _lowerBound; }
        }

        double[] _upperBound = { 100, 100 };

        /// <summary>
        /// Upper search-space boundary.
        /// </summary>
        public override double[] UpperBound
        {
            get { return _upperBound; }
        }

        /// <summary>
        /// Lower initialization boundary.
        /// </summary>
        public override double[] LowerInit
        {
            get { return LowerBound; }
        }

        /// <summary>
        /// Upper initialization boundary.
        /// </summary>
        public override double[] UpperInit
        {
            get { return UpperBound; }
        }

        /// <summary>
        /// Minimum possible fitness for this problem.
        /// </summary>
        public override double MinFitness
        {
            get { return 0; }
        }

        /// <summary>
        /// Acceptable fitness threshold.
        /// </summary>
        public override double AcceptableFitness
        {
            get { return 0.4; }
        }

        string[] _parameterName = { "a", "b" };

        /// <summary>
        /// Names of parameters for problem.
        /// </summary>
        public override string[] ParameterName
        {
            get { return _parameterName; }
        }

        /// <summary>
        /// Compute and return fitness for the given parameters.
        /// </summary>
        /// <param name="x">Candidate solution.</param>
        public override double Fitness(double[] x)
        {
            Debug.Assert(x != null && x.Length == Dimensionality);

            double a = GetA(x);
            double b = GetB(x);
            double t1 = 1 - a;
            double t2 = b - a * a;

            return t1 * t1 + 100 * t2 * t2;
        }

        /// <summary>
        /// Enforce and evaluate constraints.
        /// </summary>
        /// <param name="x">Candidate solution.</param>
        public override bool EnforceConstraints(ref double[] x)
        {
            // Enforce boundaries.
            SwarmOps.Tools.Bound(ref x, LowerBound, UpperBound);

            // Return feasibility.
            return Feasible(x);
        }

        /// <summary>
        /// Evaluate constraints.
        /// </summary>
        /// <param name="x">Candidate solution.</param>
        public override bool Feasible(double[] x)
        {
            Debug.Assert(x != null && x.Length == Dimensionality);

            double a = GetA(x);
            double b = GetB(x);

            // Radius.
            double r = Math.Sqrt(a * a + b * b);

            return ((r < 0.7) || ((r > 3) && (r < 5))) && (a < b * b);
        }
        #endregion
    }
}
