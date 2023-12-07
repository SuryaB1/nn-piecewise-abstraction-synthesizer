#define grammar: x1, real types, ≤, ≥, =
grammar_input = ["x1"]
grammar_input_temp = []
grmmar_arith_ops = ["+", "-", "*"]
grammar_bool_ops = ["==", "<=", ">="] # TODO: constrain grammar for initial implementation, and come up with corresponding model

examples = { (input) : (output) }

def get_next_expr():
    #static vars: length
    #fill grammar_input_temp with permutations of grammar_input with grmmar_arith_ops and grammar_bool_ops
    grammar_input = grammar_input_temp

def main():
    network = Marabou.read_tf(filename = filename, inputNames = inputNames, outputNames = outputName)

    start = 25
    end = -25
    curr_ex = start

    num_iter = 0
    max_iter = 100000
    while num_iter < max_iter:
        expr = get_next_expr() # returns a string based on examples and grammar

        for ex_in, ex_out in examples: 
            x1 = ex_in
            if (eval(expr) != ex_out):
                break

        if curr_ex >= end:
            x1 = curr_ex
            network.setLowerBound(input, x1) # rather than adding specific properties
            network.setUpperBound(input, x1) # TODO: look into documentation regarding expanding query to be more exhaustive, (test with )
            network.setLowerBound(output, eval(expr))
            network.setUpperBound(output, eval(expr)) # just querying Marabou about this reachability property is not enough to see if we have found a solution

            vals, stats = network.solve("marabou.log") # query Marabou regarding whether the expr's input-output property is expected from NN
            if len(vals) == 0:
                return expr
            else:
                examples.add(vals[0]) # add counter-examples
            
            curr_ex -= 1
        num_iter += 1