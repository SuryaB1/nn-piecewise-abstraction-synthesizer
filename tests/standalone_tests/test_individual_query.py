##########################################################################################################
### Constraints are as follows (where x0 and x1 refer to the two input variables, 
#   and output_var refers to the output variable):
#
# Input space constraints:
# x0 <= 100
# x0 >= -100
# x1 <= 100
# x1 >= -100
#
# Disjunctions:
# x0 <= 9.99
# x0 >= 10.01
# x1 <= 9.99
# x1 >= 10.01
#
# Inequalities:
# 1.0*x0 + 1.0*x1 <= 20.68676191034973
# 0.13542994985903067*x0 + 1.0*x1 <= 11.943228800010417
# 137.12153670362702*x0 + 1.0*x1 <= 1401.0531025403711
#
# output_var == 0
############

from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouUtils

TF_NN_FILENAME = "saved_models/diagonal-split_classif_nnet"

### Read network
network = Marabou.read_tf(filename = TF_NN_FILENAME, modelType="savedModel_v2")
inputVars = network.inputVars[0][0]
outputVars = network.outputVars[0].flatten()

network.clearProperty()

### Set bounds for both input variables
network.setLowerBound(inputVars[0], -100)
network.setUpperBound(inputVars[0], 100)
network.setLowerBound(inputVars[1], -100)
network.setUpperBound(inputVars[1], 100)

### Set bounds for the output variable
eq = MarabouUtils.Equation()
eq.addAddend(1, outputVars[0])
eq.setScalar(0.0)
network.addEquation(eq, isProperty=True)

### Add disjunction constraint
eq1 = MarabouUtils.Equation(MarabouCore.Equation.LE)
eq1.addAddend(1.0, inputVars[0])
eq1.setScalar(9.99)

eq2 = MarabouUtils.Equation(MarabouCore.Equation.GE)
eq2.addAddend(1.0, inputVars[0])
eq2.setScalar(10.01)

eq3 = MarabouUtils.Equation(MarabouCore.Equation.LE)
eq3.addAddend(1.0, inputVars[1])
eq3.setScalar(9.99)

eq4 = MarabouUtils.Equation(MarabouCore.Equation.GE)
eq4.addAddend(1.0, inputVars[1])
eq4.setScalar(10.01)

network.addDisjunctionConstraint( [[eq1], [eq2], [eq3], [eq4]] )

### Add inequality constraints
eq5 = MarabouUtils.Equation(MarabouCore.Equation.LE)
eq5.addAddend(1.0, inputVars[0])
eq5.addAddend(1.0, inputVars[1])
eq5.setScalar(20.68676191034973)
network.addEquation(eq5, isProperty=True)

eq6 = MarabouUtils.Equation(MarabouCore.Equation.LE)
eq6.addAddend(0.13542994985903067, inputVars[0])
eq6.addAddend(1.0, inputVars[1])
eq6.setScalar(11.943228800010417)
network.addEquation(eq6, isProperty=True)

eq7 = MarabouUtils.Equation(MarabouCore.Equation.LE)
eq7.addAddend(137.12153670362702, inputVars[0])
eq7.addAddend(1.0, inputVars[1])
eq7.setScalar(1401.0531025403711)
network.addEquation(eq7, isProperty=True)

### Query Marabou
options = Marabou.createOptions(verbosity = 0)
exitCode, vals, stats = network.solve(verbose=False, options=options)
print("vals:", vals)