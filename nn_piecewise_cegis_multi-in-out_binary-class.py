import itertools
from tqdm import tqdm

from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouNetwork
from maraboupy import MarabouUtils

# TF_NN_FILENAME =  "saved_models/sign_classif_nn_no-softmax" # 1D input
# TF_NN_FILENAME = "saved_models/unit-sqr_classif_nnet_no-sigmoid" # 2D input
# TF_NN_FILENAME = "saved_models/diagonal-split_classif_nnet" # 2D input, non-rectangular case
TF_NN_FILENAME = "saved_models/concave-poly_classif_nnet" # 2D input, non-rectangular case
# TF_NN_FILENAME =  "saved_models/3d-unit-sqr_classif_nnet_no-sigmoid" # 3D input
# TF_NN_FILENAME =  "saved_models/4d-unit-sqr_classif_nnet_no-sigmoid" # 4D input

DEBUG = True

SYNTHESIS_LOWER_BOUND = -100
SYNTHESIS_UPPER_BOUND = 100
DEFAULT_EPSILON_FOR_COMPARISONS = 0.0000000001
NUM_DEF_EPS_DIGITS = 10
EPSILON = 0.0001 # represents order-of-magnitude for strict greater than operation # bound values less than or equal to 0.00001 are treated as 0 by Marabou - https://github.com/NeuralNetworkVerification/Marabou/issues/666
NUM_EPS_DIGITS = 4

### To prevent overlapping boundaries and keep numbers readable as determined by EPSILON
def customRound(bound, awayFrom = None): # awayFrom is the value that the bound param must be greater than EPSILON away from (can be set to previous counterexample, or segment bounds)
    bound = round(bound, NUM_DEF_EPS_DIGITS) # to prevent, for example, 0.99999999999687 from being different from 1
    newBound = round(bound, NUM_EPS_DIGITS) # using minimum granularity of EPSILON to simplify logic so that segment bounds can be split with EPSILON as the strictly-greater-than threshold --> otherwise, would need extra logic if the current counter example was within EPSILON of any of the segment boundaries (in other words, approximating segment boundaries to nearest EPSILON)
    if newBound != bound: # newBound == awayFrom: # and bound != awayFrom: # (newBound != bound:, for when awayFrom not close to bound, like within 0 and eps in 2D case) # prioritize outward expansion (rounding away from bounds and awayFrom) rather than inward expansion (rounding towards them), since Marabou returns counter-examples on border, and do not want to, for example, round down to input coord on the border with diff output value than outside border
        # in other words, (if awayFrom = prev counterex), make sure the bound rounds away from awayFrom
        if bound > awayFrom:
            newBound += EPSILON 
        elif bound < awayFrom:
            newBound -= EPSILON
    return newBound

def form_query(network, outputVarIdx, curr_segment):
    inputVars = network.inputVars[0][0]
    if (DEBUG): print("_query_")
    for var in inputVars:
        if (DEBUG): print("input var lower bound:", curr_segment[0][var], end = ", ")
        network.setLowerBound(var, curr_segment[0][var])
        if (DEBUG): print("input var upper bound:", curr_segment[1][var])
        network.setUpperBound(var, curr_segment[1][var])
    
    if curr_segment[2]:
        if (DEBUG): print(f"output var {outputVarIdx - len(inputVars)} upper bound:", 0)
        network.setUpperBound(outputVarIdx, 0)
    else:
        if (DEBUG): print(f"output var {outputVarIdx - len(inputVars)} lower bound:", EPSILON)
        network.setLowerBound(outputVarIdx, EPSILON)

def main():
    network = Marabou.read_tf(filename = TF_NN_FILENAME, modelType="savedModel_v2")
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0].flatten()

    nn_abstr = dict() # full piecewise abstraction for input neural network
    
    # Loop 1: Synthesize piecewise mapping for each output variable
    for i in tqdm(range(len(outputVars))):
        if (DEBUG): print(f"--- Searching piecewise mapping for Output Variable {i} ---")
        outputVarIdx = outputVars[i]

        initial_lower_bound = list(SYNTHESIS_LOWER_BOUND for idx in inputVars)
        initial_upper_bound = list(SYNTHESIS_UPPER_BOUND for idx in inputVars)

        stack = [[initial_lower_bound, initial_upper_bound, 0]] # stack of unvisited segments # new (multi-in): [[lower], [upper], val] # old (single-in): { lower : [upper, val] }
        mappings = [] # finalized mappings for current output variable
            
        set_not_split = True # set first then split to avoid unnecessary segments
        prev_cntr_ex = None # needed to assign output values to new segments when splitting
        ### Loop 2: Find and resolve all segments
        ### Loop 3: Get next correct segment (by repeatedly querying and correcting next segment in consideration)
        while stack:
            curr_segment = stack.pop() # get next candidate mapping
            if (DEBUG): print("current segment = ", curr_segment)

            if (DEBUG): print("stack:", stack)
            if (DEBUG): print("mappings so far:", mappings)
            
            network.clearProperty()
            form_query(network, outputVarIdx, curr_segment)
            options = Marabou.createOptions(snc=False)
            # if (DEBUG): print(network.getMarabouQuery())
            if (DEBUG): print("start query")
            exitCode, vals, stats = network.solve(verbose=False, options=options) # filename="marabou.log", <-- Submit issue # query Marabou regarding whether the expr's input-output property is expected from NN
            if (DEBUG): print("vals", vals)
            
            ### Get next sub-candidate mapping
            if exitCode == "sat":
                if set_not_split:
                    # First, attempt setting the entire segment's output value to the counterexample's output value
                    if (DEBUG): print("SET")
                    curr_segment[2] = int(vals[outputVarIdx] > 0)
                    prev_cntr_ex = vals
                    stack.append(curr_segment) # Add set segment to stack
                else:
                    # After attempting to set the output value of a segment, split the segment based on both the new and old counter-example
                    if (DEBUG): print("SPLIT")

                    # Compute representation of deviation between new counterexample and previous counterexample (during set attempt) to later determine how to assign split segments' output value(s)
                    split_idxs = []
                    prev_counterex_relative_segment = [] # For example, in 2D input case, [0, 0] is bottom-left quadrant, [0, 1] is top-left, [1, 0] is bottom-right, and [1, 1] is top-right
                    prev_curr_counterex_delta_signs = [] # ALIGNED_EDGE_CASE
                    for key in inputVars:
                        value = customRound(vals[key], awayFrom = prev_cntr_ex[key])
                        split_idxs.append(value)

                        diff = prev_cntr_ex[key] - value
                        prev_counterex_relative_segment.append(int(diff > 0))

                        delta_signs = int(diff / abs(diff)) if diff != 0 else 0 # ALIGNED_EDGE_CASE
                        prev_curr_counterex_delta_signs.append(delta_signs) # ALIGNED_EDGE_CASE
                    isAlignedCase = (len(prev_curr_counterex_delta_signs) - prev_curr_counterex_delta_signs.count(0)) == 1 # ALIGNED_EDGE_CASE

                    # Compute the bounds that define each split segment, and add to stack
                    bound_options = [curr_segment[0], split_idxs, curr_segment[1]]
                    if (DEBUG): print("bound_options:", bound_options)
                    for segment in itertools.product(*[[0, 1] for idx in range(len(inputVars))]): # For example, in 2D input case, (0, 0) is bottom-left quadrant, (0, 1) is top-left, (1, 0) is bottom-right, and (1, 1) is top-right
                        bottom_left = []
                        top_right = []
                        alignedCasePrevCounterExSegment = [] # ALIGNED_EDGE_CASE
                        skipSegment = False

                        # Get upper bound of split's lower segment and lower bound of upper segment
                        for dim_idx, dim_val in enumerate(segment): # Note: next_segment_offset only affects particular dimensions of different bounds of different segments
                            if (isAlignedCase): alignedCasePrevCounterExSegment.append(min(prev_curr_counterex_delta_signs[dim_idx] + 1, 1)) # ALIGNED_EDGE_CASE
                            
                            # Determines if there is an offset for this dimension's value or not
                            curr_dim_offset = EPSILON * int(dim_val == prev_counterex_relative_segment[dim_idx]) # offset (gap) only when current dim of current segment's lower bound is same as corresponding dim of lower bound of segment containing prev counter_ex
                            
                            # Determines where the offset (inter-split gaps) should be and in what direction
                            bottom_left_offset = int(dim_val == 1) * curr_dim_offset
                            top_right_offset = -1 * int(dim_val == 0) * curr_dim_offset
                            if (DEBUG): print("bottom_left_offset:", bottom_left_offset)
                            if (DEBUG): print("top_right_offset:", top_right_offset)

                            bottom_left_curr_dim_val = round(bound_options[dim_val][dim_idx] + bottom_left_offset, NUM_EPS_DIGITS)
                            top_right_curr_dim_val = round(bound_options[dim_val + 1][dim_idx] + top_right_offset, NUM_EPS_DIGITS)
                            if bottom_left_curr_dim_val > bound_options[2][dim_idx] or bottom_left_curr_dim_val < bound_options[0][dim_idx] \
                                or top_right_curr_dim_val > bound_options[2][dim_idx] or top_right_curr_dim_val < bound_options[0][dim_idx]:
                                skipSegment = True
                                break
                            
                            # Compile bottom left and top right bounds for this new split segment
                            # Why round? - # https://stackoverflow.com/questions/11873046/python-weird-addition-bug
                            bottom_left.append( bottom_left_curr_dim_val )
                            top_right.append( top_right_curr_dim_val )
                            if (DEBUG): print("bottom_left:", bottom_left)
                            if (DEBUG): print("top_right:", top_right)
                        
                        if skipSegment:
                            continue

                        usePrevCounterExOutput = (list(segment) == alignedCasePrevCounterExSegment) if isAlignedCase else (list(segment) == prev_counterex_relative_segment) # ALIGNED_EDGE_CASE
                        val = int( ((prev_cntr_ex[outputVarIdx] > 0) ^ (not usePrevCounterExOutput)) ) # int(prev_cntr_ex[outputVarIdx] > 0) if usePrevCounterExOutput else int(prev_cntr_ex[outputVarIdx] <= 0) # int( ((prev_cntr_ex[outputVarIdx] > 0) ^ (not usePrevCounterExOutput)) )
                        new_segment = [bottom_left, top_right, val]
                        if (DEBUG): print(f"a split segment: {new_segment}")
                        stack.append(new_segment) # Keep track of segment

                set_not_split = not set_not_split
            else:
                mappings.append(curr_segment)
                set_not_split = 1
        nn_abstr[i] = mappings
    print("Mappings:", nn_abstr)

if __name__ == '__main__':
    main()