import qopt

# COMBINATORIAL #
import _knapsack
import _func1d

knapsack = _knapsack.KnapsackProblem(qopt.path('problems/knapsack/knapsack-250.txt'))
knapsack10 = _knapsack.KnapsackProblem(qopt.path('problems/knapsack/knapsack-10.txt'))
knapsack100 = _knapsack.KnapsackProblem(qopt.path('problems/knapsack/knapsack-100.txt'))
knapsack500 = _knapsack.KnapsackProblem(qopt.path('problems/knapsack/knapsack-500.txt'))

# func1d
class Funcs: pass
func1d = Funcs()
func1d.f1 = _func1d.Func1D(1)
func1d.f2 = _func1d.Func1D(2)
func1d.f3 = _func1d.Func1D(3)

# sat


# REAL #

# cec2005
import _cec2005
cec2005 = _cec2005.CEC2005

# cec2011
# if os.environ('LD_LIBRARY_PATH').contains('.../CEC2011') and ...('Matlab2011..  # <- TODO
import _cec2011
cec2011 = _cec2011.CEC2011


