import itertools
from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouNetwork
from maraboupy import MarabouUtils

TF_NN_FILENAME = "../../data/inputs/models/saved_models/three_in_two_out_nn" # "basic_nn" # "reluplex_paper_fig2_nn"
SCALAR_BOUND = 10

def get_next_candidate(network, comb, outputVarIdx):
    eq_str = "0" # to reduce if-statement checks and in case start coefficient is negative and fails to execute
    # eq = MarabouUtils.Equation(MarabouCore.Equation.EquationType.EQ)
    eq_le = MarabouCore.Equation(MarabouCore.Equation.EquationType.LE) # MarabouUtils.Equation(MarabouCore.Equation.EquationType.LE)
    eq_ge = MarabouCore.Equation(MarabouCore.Equation.EquationType.GE)

    inputVars = network.inputVars[0][0]
    for var in inputVars:
        coeff = comb[var]
        # eq.addAddend(coeff, var)
        eq_le.addAddend(coeff, var)
        eq_ge.addAddend(coeff, var)

        # eq_str += f"{coeff}*x{idx}" if not idx or coeff < 0 else f"+{coeff}*x{idx}"
        eq_str += f"+({coeff})*x{var}"

        # add additional bounds for variables
        network.setLowerBound(var, 0) # network.setLowerBound(var, -1*SCALAR_BOUND)
        network.setUpperBound(var, SCALAR_BOUND)
    # eq.addAddend(-1, outputVarIdx)
    eq_le.addAddend(-1, outputVarIdx)
    eq_ge.addAddend(-1, outputVarIdx)

    scalar = comb[len(inputVars)]
    # eq.setScalar(scalar)
    eq_le.setScalar(scalar-1)
    eq_ge.setScalar(scalar+1)
    eq_str += f"+({-1*scalar})"

    return eq_le, eq_ge, eq_str

def main():
    network = Marabou.read_tf(filename = TF_NN_FILENAME, modelType="savedModel_v2")
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0].flatten()

    expressions = []
    # print(outputVars)
    examples = []
    for i in range(len(outputVars)):
        scalar_vals = [[i for i in range (-1*SCALAR_BOUND, SCALAR_BOUND+1)] for i in range(len(inputVars) + 1)] # Grammar based on Theory of Integers
        scalar_combs = itertools.product(*scalar_vals)

        print("-------------")
        print("outputVar:", i)
        outputVarIdx = outputVars[i]
        for comb in scalar_combs:
            network.clearProperty()
            satisfies_examples = True

            cand_le, cand_ge, cand_str = get_next_candidate(network, comb, outputVarIdx)
            print(cand_str)

            for ex in examples: 
                ex_in = []
                for in_idx in inputVars:
                    ex_in.append(ex[in_idx])
                    exec(f"x{in_idx} = {ex[in_idx]}")
                print(eval(cand_str))
                if (eval(cand_str) != ex[outputVarIdx]):
                    print("existing counterexample:", ex_in, "ex_out:", ex[outputVarIdx])
                    satisfies_examples = False
                    break

            if satisfies_examples:
                # print(cand_str)
                disjunction = [[cand_le], [cand_ge]]
                network.addDisjunctionConstraint(disjunction)
                # MarabouCore.addDisjunctionConstraint(inputQuery=network.getMarabouQuery(), disjuncts=disjunction)
                # MarabouNetwork.addDisjunctionConstraint(network, [[cand_le], [cand_ge]])

                # print("query:", network.getMarabouQuery().getNumOutputVariables())
                exitCode, vals, stats = network.solve("marabou.log") # query Marabou regarding whether the expr's input-output property is expected from NN
                # print("exit code:", exitCode)
                print("vals:", vals) # indices is variable indices, value is satisfying assignment
                if len(vals) == 0:
                    # print(cand_str)
                    print("---------------------------------")
                    expressions.append(cand_str)
                    break
                else:
                    examples.append(vals) # add counter-examples
        if len(expressions) == i:
            expressions.append(None)
    print(expressions)

if __name__ == '__main__':
    main()