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
from maraboupy import Marabou # Came across this case when attempting to address minor bound change non-terminating query issue by rounding bounds

TF_NN_FILENAME = "saved_models/sign_classif_nn_no-softmax"

### Read network
network = Marabou.read_tf(filename = TF_NN_FILENAME, modelType="savedModel_v2")
inputVars = network.inputVars[0][0]
outputVars = network.outputVars[0].flatten()

### Set query's input variable bounds
network.setLowerBound(inputVars[0], 0.000011)
network.setUpperBound(inputVars[0], 1000.0)

### Set query's output variable bounds
upperBound = 0
network.setUpperBound(outputVars[0], upperBound)

### Query Marabou
options = Marabou.createOptions(snc=False)
exitCode, vals, stats = network.solve(verbose=False, options=options)
print("vals:", vals)