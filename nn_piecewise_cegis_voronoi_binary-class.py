from tqdm import tqdm
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
from random import randint
from dataclasses import dataclass
import ast

from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouNetwork
from maraboupy import MarabouUtils

# Test cases
# TF_NN_FILENAME = "saved_models/sign_classif_nn_no-softmax" # 1D input
# TF_NN_FILENAME = "saved_models/unit-sqr_classif_nnet_no-sigmoid" # 2D input
TF_NN_FILENAME = "saved_models/diagonal-split_classif_nnet" # 2D input, non-rectangular case
# TF_NN_FILENAME = "saved_models/concave-poly_classif_nnet" # 2D input, non-rectangular case
# TF_NN_FILENAME = "saved_models/3d-unit-sqr_classif_nnet_no-sigmoid" # 3D input
# TF_NN_FILENAME = "saved_models/4d-unit-sqr_classif_nnet_no-sigmoid" # 4D input

DEBUG = True

# Bounds on all axes of input space for which piecewise mapping is synthesized
SYNTHESIS_LOWER_BOUND = -100
SYNTHESIS_UPPER_BOUND = 100

# Counterexample handling states
SET_STATE = 0
SPLIT_STATE = 1

@dataclass
class Hyperplane:
    coeffs: list # list of floats
    constant: float
    type: MarabouCore.Equation.EquationType

class VoronoiCell:
    input_dim = 2

    def __init__(self, centroid, ridges, output, neighbors=[]):
        self.centroid = centroid
        self.ridges = ridges
        self.output = output
        self.neighbors = neighbors
    
    @staticmethod
    def compute_voronoi_tessellation(input_output_dict, output_var_idx):
        # Obtain Voronoi tessellation
        new_centroids = list(input_output_dict.keys())
        vor = Voronoi(new_centroids)
        bounded_ridges = [ridge for ridge in vor.ridges if not -1 in ridge]

        # Construct new cells from generated centroids and Voronoi tessellation
        new_cells = []
        for i in range(len(vor.points)):
            centr = vor.points[i]
            ridges = VoronoiCell.get_ridges_for_region(centr, vor.regions[i], bounded_ridges, vor.vertices)
            new_cells.append(VoronoiCell(centr, ridges, input_output_dict[centr]))

        # TODO: populate .neighbors with O(n^2) for loop comparing ridges and search for match

        return new_cells
    
    @staticmethod
    def get_ridges_for_region(centroid, region, bounded_ridges, vertices): # Given a region and a set of bounded regions, return the ridges of this region as Hyperplane objects
        region = set(region)
        ridges_of_given_region = []
        for ridge in bounded_ridges:
            if set(ridge).issubset(region):
                ridges_of_given_region.append(VoronoiCell.get_hyperplane(centroid, ridge, vertices))
        return ridges_of_given_region

    @staticmethod
    def get_hyperplane(centroid, ridge, vertices): # Given a ridge, return it as a hyperplane by solving Ax=b (where x is the coefficients) using least squares
        A = np.array(vertices[ridge]) # Array of vertices
        b = np.ones(A.shape[0])

        coefficients = np.linalg.pinv(A) @ b # TODO: compare with np.linalg.lstsq for computing hyperplane from vertices
        constant = 1
        
        inequality_type = None
        if centroid dot coeffs < constant: 
            inequality_type = MarabouCore.Equation.EquationType.LE
        elif centroid dot coeffs < constant:
            inequality_type = MarabouCore.Equation.EquationType.GE
        else:
            raise ValueError("Centroid lies on VoronoiCell ridge")
    
        return Hyperplane(coeffs, constant, inequality_type)

def debug_log(*str):
    if DEBUG:
        print(str[0])

# Assumes input data is flattened to single dimensional Python list, and output class is integer
def read_init_datapoints(filename="", input_dim=0, output_dim=0):
    if filename == "": return dict()

    file = open(filename, 'r')
    lines = file.readlines()

    input_output_dict = dict()
    for line in lines:
        split_line = line.split(':')

        input_coord = tuple([float(element.strip()) for element in ast.literal_eval(split_line[0].strip())])
        assert len(input_coord) == input_dim

        output = tuple([element.strip() for element in ast.literal_eval(split_line[1].strip())])
        assert len(output) == output_dim

        input_output_dict[input_coord] = output
    return input_output_dict

def construct_initial_ridge_vertices(input_vars):
    init_ridges = []
    for var in input_vars:
        temp_coeffs = [0 if i != var else 1 for i in input_vars]
        init_ridges.append(Hyperplane(temp_coeffs, SYNTHESIS_UPPER_BOUND, MarabouCore.Equation.EquationType.LE))
        init_ridges.append(Hyperplane(temp_coeffs, SYNTHESIS_LOWER_BOUND, MarabouCore.Equation.EquationType.GE))
    return init_ridges

def form_query(network, output_var, curr_segment):
    # Add inequalities for synthesis bounds to query (Note: not checking for when curr_segment.ridges length < network.inputVars[0][0], since segment could still be open)
    for var in network.inputVars[0][0]:
        network.setLowerBound(var, SYNTHESIS_LOWER_BOUND)
        network.setUpperBound(var, SYNTHESIS_UPPER_BOUND)

    # Add inequalities for current segment (the segment currently being searched for counterexamples)
    for ridge in curr_segment.ridges:
        eq = MarabouCore.Equation(ridge.type)

        input_vars = network.input_vars[0][0]
        for var in input_vars:
            eq.addAddend(ridge.coeffs[var], var)
        eq.setconstant(ridge.constant)

        network.addEquation(eq)

    # Add equality expressing class assignment for current segments
    eq = MarabouCore.Equation(MarabouCore.Equation.EquationType.EQ)
    eq.addAddend(1, output_var)
    eq.setconstant(ridge.output)
    network.addEquation(eq)

def main():
    network = Marabou.read_tf(filename = TF_NN_FILENAME, modelType="savedModel_v2")
    input_vars = network.inputVars[0][0]
    output_vars = network.outputVars[0].flatten()
    VoronoiCell.input_dim = len(input_vars)
    init_datapoints = read_init_datapoints("diagonal-split_classif_nnet_sample_datapoints.txt", len(input_vars), len(output_vars))
    
    counterexample_handling_state = SET_STATE
    # Loop 1: Synthesize piecewise mapping for each output variable (this algorithm assumes each output is an integer)
    for i in tqdm(range(len(output_vars))):
        debug_log(f"--- Searching piecewise mapping for Output Variable {i} ---")
        output_var = output_vars[i]

        centroid_cell_map = dict() # Dictionary of all searched centroids in input space mapped to corresponding VoronoiCell objects
        incomplete_segments = [] # Stack of centroids of segments left to search for counterexamples
        
        if init_datapoints:
            incomplete_segments.extend(VoronoiCell.compute_voronoi_tessellation(init_datapoints, i))
        else: # if no user-given initial datapoint, set input space as the initial Voronoi cell
            init_centroid = tuple([ (SYNTHESIS_LOWER_BOUND + SYNTHESIS_UPPER_BOUND) / 2 for i in range(len(input_vars)) ])
            initial_cell = VoronoiCell(init_centroid, construct_initial_ridge_vertices(input_vars), 0)

            centroid_cell_map[init_centroid] = initial_cell
            incomplete_segments.append(initial_cell)

        while incomplete_segments:
            curr_segment = incomplete_segments.pop()

            network.clearProperty()
            form_query(network, output_var, centroid_cell_map[curr_segment]) # Use current segment inequalities on Marabou
            exitCode, vals, stats = network.solve(verbose=False) # Query Marabou for counterexamples within segment

            if len(vals):
                if counterexample_handling_state == SET_STATE:
                    centroid_cell_map[curr_segment].output = int(vals[output_var])
                elif counterexample_handling_state == SPLIT_STATE:
                    new_centroid = tuple([float(vals[var]) for var in input_vars])
                    # centroid_cell_map.keys() + new_centroid for regenerating voronoi tessellation
                    incomplete_segments.append(new_centroid)
                    # TODO: see line 20 (need to merge Voronoi tessellation of new centroid with non-uniform boundary of existing segment)


if __name__ == '__main__':
    main()