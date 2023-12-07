from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouNetwork
from maraboupy import MarabouUtils

TF_NN_FILENAME = "saved_models/sign_classif_nn_no-softmax"

SYNTHESIS_LOWER_BOUND = -1000.0
SYNTHESIS_UPPER_BOUND = 1000.0
# TODO: Documentation
DEFAULT_EPSILON_FOR_COMPARISONS = 0.0000000001 # Taken from value of GlobalConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS Marabou implementation variable
EPSILON = 0.0001 #0.000011 # 0.00001 # epsilons less than or equal to 0.00001 cause very long query time for unsat case
# THRESH = 0.5 * EPSILON # 0.000005 # Cannot adjust inputVar bounds to be less than EPSILON by a lesser THRESH amount --> there seems to be precision correlation between input and output bounds
# 0.000005 <-- running with EPS = this
def customRound(bound):
    # dist_from_eps = bound - EPSILON
    # if abs(dist_from_eps) < DEFAULT_EPSILON_FOR_COMPARISONS:
    #     if dist_from_eps > 0:
    #         bound = EPSILON + THRESH #1.5*EPSILON # EPSILON + DEFAULT_EPSILON_FOR_COMPARISONS
    #     elif dist_from_eps < 0:
    #         bound = EPSILON - THRESH #0.5*EPSILON # EPSILON - DEFAULT_EPSILON_FOR_COMPARISONS
    bound = round(bound, 10) # make sure anything close enough to EPSILON (as per Marabou) equals EPSILON—to avoid situations with new segment's upper bounds being less than lower bound
    if bound > 0 and bound < EPSILON: # to avoid getting counter-examples within 0 and EPSILON that is small enough to cause non-terminating query issues
        return 0
    return bound #return round(bound, 4) # float("{:.9f}".format(bound)) # float("{:.4f}".format(bound))
    # return float("{:.29f}".format(bound))

def form_query(network, outputVarIdx, lower_upper_val_dict, curr_lower_bound):
    inputVars = network.inputVars[0][0]
    print("_query_")
    for var in inputVars:
        print("input var lower bound:", curr_lower_bound[var], end = ", ")
        network.setLowerBound(var, curr_lower_bound[var])
        print("input var upper bound:", lower_upper_val_dict[curr_lower_bound][0][var])
        network.setUpperBound(var, lower_upper_val_dict[curr_lower_bound][0][var])
    
    outputVal = lower_upper_val_dict[curr_lower_bound][1]
    if outputVal:
        network.setUpperBound(outputVarIdx, 0)
    else:
        network.setLowerBound(outputVarIdx, EPSILON)

def main():
    network = Marabou.read_tf(filename = TF_NN_FILENAME, modelType="savedModel_v2")
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0].flatten()

    mappings = []
    # Loop 1: Synthesize piecewise mapping for each output variable
    for i in range(len(outputVars)):
        print(f"--- Searching piecewise mapping for Output Variable {i} ---")
        outputVarIdx = outputVars[i]
        curr_lower_bound = tuple(SYNTHESIS_LOWER_BOUND for i in inputVars)
        next_segment_lower_bound = tuple(SYNTHESIS_UPPER_BOUND for i in inputVars)
        lower_upper_val_dict = {curr_lower_bound: [next_segment_lower_bound, 0]} # { lower : [upper, val] }

        ### Loop 2: Find and resolve all segments
        while True:
            set_not_split = True # set first then split to avoid unnecessary segments
            prev_cntr_ex = None # needed to assign output values to new segments when splitting

            ### Loop 3: Get next correct segment (by repeatedly querying and correcting next segment in consideration)
            while True: #exitCode == "sat": # because cannot get smallest counterexamples (would make more efficient, if possible)
                print("mapping:", lower_upper_val_dict)
                
                network.clearProperty()
                form_query(network, outputVarIdx, lower_upper_val_dict, curr_lower_bound)
                options = Marabou.createOptions(snc=False)
                # print(network.getMarabouQuery())
                print("start query")
                exitCode, vals, stats = network.solve(verbose=False, options=options) # filename="marabou.log", # query Marabou regarding whether the expr's input-output property is expected from NN
                print("vals", vals)
                
                ### Get next sub-candidate mapping
                if exitCode == "sat":
                    if set_not_split:
                        # First, attempt setting the entire segment's output value to the counterexample's output value
                        print("SET")
                        lower_upper_val_dict[curr_lower_bound][1] = int(vals[outputVarIdx] > 0)
                        prev_cntr_ex = vals
                    else:
                        # After attempting to set the output value of a segment, split the segment based on both the new and old counter-example
                        print("SPLIT")

                        # Get difference between new counterexample and previous counterexample (during set attempt) to later determine how to assign split segments' output value(s)
                        curr_prev_counterex_diff = 0
                        for key, value in vals.items():
                            curr_prev_counterex_diff = value - prev_cntr_ex[key]
                            if curr_prev_counterex_diff != 0:
                                break
                            # diff = value - prev_cntr_ex[key]
                            # if curr_prev_counterex_diff == 0 and diff != 0:
                            #     curr_prev_counterex_diff = diff
                            # elif curr_prev_counterex_diff != 0:
                            #     break
                        
                        # Get upper bound of split's lower segment and lower bound of upper segment
                        split_idxs = []
                        next_segment_lower_bound = []
                        if curr_prev_counterex_diff < 0:
                            for key, value in vals.items():
                                if key < len(inputVars):
                                    split_idxs.append(customRound(value))
                                    next_segment_lower_bound.append(split_idxs[-1] + EPSILON)
                                else:
                                    break
                        else:
                            for key, value in vals.items():
                                if key < len(inputVars):
                                    next_segment_lower_bound.append(customRound(value))
                                    split_idxs.append(next_segment_lower_bound[-1] - EPSILON)
                                else:
                                    break
                        
                        # Split the segment
                        next_segment_lower_bound = tuple(next_segment_lower_bound)
                        lower_upper_val_dict[next_segment_lower_bound] = [lower_upper_val_dict[curr_lower_bound][0], int( (vals[outputVarIdx] > 0) ^ (curr_prev_counterex_diff < 0) )] # new upper segment
                        lower_upper_val_dict[curr_lower_bound] = [tuple(split_idxs), int(not( (vals[outputVarIdx] > 0) ^ (curr_prev_counterex_diff < 0) ))] # new lower segment
                    set_not_split = not set_not_split
                else:
                    break

            ### Get next candidate mapping
            # print("before")
            # print(curr_lower_bound)
            # print(next_segment_lower_bound)
            if len(next_segment_lower_bound) == 0: # more efficient than lower_upper_val_dict[curr_lower_bound][0] == 1000
                break
            curr_lower_bound = next_segment_lower_bound
            next_segment_lower_bound = tuple(i + EPSILON for i in lower_upper_val_dict[next_segment_lower_bound][0] if i < SYNTHESIS_UPPER_BOUND) # the next lower bound if next segment search ends on SET state
        
        mappings.append(lower_upper_val_dict)
    print("Mappings:", mappings)

if __name__ == '__main__':
    main()