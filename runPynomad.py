import sys
import PyNomad
import bbPynomad

lb      = eval(sys.argv[1])
ub      = eval(sys.argv[2])
params  = eval(sys.argv[3])

sol = PyNomad.optimize(bbPynomad.bbPynomad, [], lb, ub, params)
print(sol)