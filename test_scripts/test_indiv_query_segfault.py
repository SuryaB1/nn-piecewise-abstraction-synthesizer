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
eq1.setScalar(-76.49109338285359)

eq2 = MarabouUtils.Equation(MarabouCore.Equation.GE)
eq2.addAddend(1.0, inputVars[0])
eq2.setScalar(-76.47109338285358)

eq3 = MarabouUtils.Equation(MarabouCore.Equation.LE)
eq3.addAddend(1.0, inputVars[1])
eq3.setScalar(-75.49109338285359)

eq4 = MarabouUtils.Equation(MarabouCore.Equation.GE)
eq4.addAddend(1.0, inputVars[1])
eq4.setScalar(-75.47109338285358)

network.addDisjunctionConstraint( [[eq1], [eq2]] )
network.addDisjunctionConstraint( [[eq3], [eq4]] )

### Add inequality constraints
eq5 = MarabouUtils.Equation(MarabouCore.Equation.LE)
eq5.addAddend(1.0, inputVars[0])
eq5.addAddend(1.0, inputVars[1])
eq5.setScalar(-151.00762263177126)
network.addEquation(eq5, isProperty=True)

eq6 = MarabouUtils.Equation(MarabouCore.Equation.LE)
eq6.addAddend(0.8083398365828554, inputVars[0])
eq6.addAddend(1.0, inputVars[1])
eq6.setScalar(-68.28943321943842)
network.addEquation(eq6, isProperty=True)

eq7 = MarabouUtils.Equation(MarabouCore.Equation.GE)
eq7.addAddend(0.256483128499954, inputVars[0])
eq7.addAddend(1.0, inputVars[1])
eq7.setScalar(-95.81392131329568)
network.addEquation(eq7, isProperty=True)

eq8 = MarabouUtils.Equation(MarabouCore.Equation.GE)
eq8.addAddend(1.0, inputVars[0])
eq8.addAddend(1.0, inputVars[1])
eq8.setScalar(-152.73014757790332)
network.addEquation(eq8, isProperty=True)

eq9 = MarabouUtils.Equation(MarabouCore.Equation.LE)
eq9.addAddend(7.976596849687285, inputVars[0])
eq9.addAddend(1.0, inputVars[1])
eq9.setScalar(-678.1605099595745)
network.addEquation(eq9, isProperty=True)

### Query Marabou
options = Marabou.createOptions(verbosity = 0)
exitCode, vals, stats = network.solve(verbose=False, options=options)
print("vals:", vals)