# from maraboupy import Marabou

# TF_NN_FILENAME = "saved_models/sign_classif_nn_no-softmax"

# network = Marabou.read_tf(filename = TF_NN_FILENAME, modelType="savedModel_v2")
# inputVars = network.inputVars[0][0]
# outputVars = network.outputVars[0].flatten()

# network.clearProperty()

# network.setLowerBound(inputVars[0], -999.999995)
# network.setUpperBound(inputVars[0], 1000.0) # works when = 1000.0

# lowerBound = 0.000005 # works when greater than current value
# network.setLowerBound(outputVars[0], lowerBound) # works when lowerBound is equal to 0 --> seems to be some sort of correlation between input and output bound precision

# options = Marabou.createOptions(snc=False)
# print("start query")
# exitCode, vals, stats = network.solve(verbose=False, options=options)
# print("vals:", vals)

# from maraboupy import Marabou # Came across this case when attempting to address minor bound change non-terminating query issue by rounding bounds

# TF_NN_FILENAME = "saved_models/sign_classif_nn_no-softmax"

# ### Read network
# network = Marabou.read_tf(filename = TF_NN_FILENAME, modelType="savedModel_v2")
# inputVars = network.inputVars[0][0]
# outputVars = network.outputVars[0].flatten()

# ### Set query's input variable bounds
# network.setLowerBound(inputVars[0], 0.000011)
# network.setUpperBound(inputVars[0], 1000.0)

# ### Set query's output variable bounds
# upperBound = 0
# network.setUpperBound(outputVars[0], upperBound)

# ### Query Marabou
# options = Marabou.createOptions(snc=False)
# exitCode, vals, stats = network.solve(verbose=False, options=options)
# print("vals:", vals)

##########################################################################################################
### To summarize, the constraints are as follows (where x0 and x1 refer to the two input variables, 
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
# 1.0*x0 + 1.0*x1 >= 19.986117936461756
# -2.1425049404379113*x0 + 1.0*x1 <= -10.535579455125532
# -1.2074084055251617*x0 + 1.0*x1 >= -2.222228741939958
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

eq8 = MarabouUtils.Equation(MarabouCore.Equation.GE)
eq8.addAddend(1.0, inputVars[0])
eq8.addAddend(1.0, inputVars[1])
eq8.setScalar(19.986117936461756)
network.addEquation(eq8, isProperty=True)

eq9 = MarabouUtils.Equation(MarabouCore.Equation.LE)
eq9.addAddend(-2.1425049404379113, inputVars[0])
eq9.addAddend(1.0, inputVars[1])
eq9.setScalar(-10.535579455125532)
network.addEquation(eq9, isProperty=True)

eq10 = MarabouUtils.Equation(MarabouCore.Equation.GE)
eq10.addAddend(-1.2074084055251617, inputVars[0])
eq10.addAddend(1.0, inputVars[1])
eq10.setScalar(-2.222228741939958)
network.addEquation(eq10, isProperty=True)

### Query Marabou
options = Marabou.createOptions(verbosity = 0)
exitCode, vals, stats = network.solve(verbose=False, options=options)
print("vals:", vals)