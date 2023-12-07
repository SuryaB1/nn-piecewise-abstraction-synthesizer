import itertools
from maraboupy import Marabou
from maraboupy import MarabouCore

filename = "reluplex_fig2_nnet"
scalar_bound = 50

def get_next_candidate(network, comb): # note: can also use sympy for encoding mathematical expressions
    global scalar_comb_idx
    scalar_comb_idx = 0
    
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0]

    eq_str = ""
    eq = MarabouCore.Equation(MarabouCore.Equation.EquationType.EQ) # MarabouCore.Equation.EquationType.EQ

    for idx, var in enumerate(inputVars): # TODO: try var as idx after resolving errors
        coeff = comb[idx]
        eq.addAddend(coeff, var)
        eq_str += f"{coeff}*x{idx}" if not idx or coeff < 0 else f"+{coeff}*x{idx}" # TODO: make not equals

        # add additional bounds for the variable
        # net1.setLowerBound(var, 0)
        # net1.setUpperBound(var, 1)
    eq.setScalar(outputVars[0])

    return eq, eq_str

def main():
    network = Marabou.read_tf(filename = filename, modelType="savedModel_v2")

    examples = dict()

    scalar_vals = [[i for i in range (-1*scalar_bound, scalar_bound+1)] for i in range(len(network.inputVars))]
    scalar_combs = itertools.product(*scalar_vals)

    for comb in scalar_combs:
        network.clearProperty()
        satisfies_examples = True

        candidate, candidate_str = get_next_candidate(network, comb)
        # print(candidate_str)

        for ex_in, ex_out in examples: 
            for in_idx in range(len(ex_in)):
                exec(f"x{in_idx} = {ex_in[in_idx]}")

            if (eval(candidate_str) != ex_out):
                satisfies_examples = False
                break

        if satisfies_examples:
            network.addEquation(candidate, isProperty=True)
            print(candidate.EquationType)
            vals, stats = network.solve("marabou.log") # query Marabou regarding whether the expr's input-output property is expected from NN
            if len(vals) == 0:
                print(candidate_str)
                break
            else:
                examples.update(vals) # add counter-examples
        
        num_iter += 1
    print("# of iterations:", num_iter)

if __name__ == '__main__':
    main()