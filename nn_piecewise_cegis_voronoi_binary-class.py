from tqdm import tqdm
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
from random import randint
from dataclasses import dataclass

from maraboupy import Marabou
from maraboupy import MarabouCore
from maraboupy import MarabouNetwork
from maraboupy import MarabouUtils

# Test cases
# TF_NN_FILENAME =  "saved_models/sign_classif_nn_no-softmax" # 1D input
# TF_NN_FILENAME = "saved_models/unit-sqr_classif_nnet_no-sigmoid" # 2D input
TF_NN_FILENAME = "saved_models/diagonal-split_classif_nnet" # 2D input, non-rectangular case
# TF_NN_FILENAME = "saved_models/concave-poly_classif_nnet" # 2D input, non-rectangular case
# TF_NN_FILENAME =  "saved_models/3d-unit-sqr_classif_nnet_no-sigmoid" # 3D input
# TF_NN_FILENAME =  "saved_models/4d-unit-sqr_classif_nnet_no-sigmoid" # 4D input

# TODO: look into utilities for obtaining coeffs and constants as and for using these coeffs and constants for to generate random points

DEBUG = True

SYNTHESIS_LOWER_BOUND = -100
SYNTHESIS_UPPER_BOUND = 100

@dataclass
class Hyperplane:
    coeffs: list
    constant: float
    type: MarabouCore.Equation.EquationType

class VoronoiCell:
    input_dim = 2

    def __init__(self, centroid, ridges, output):
        self.centroid = centroid
        self.ridges = ridges
        self.output = output

    # Returns list of VoronoiCells from scatter inference on current VoronoiCell
    def scatter_inference(self, num_points_min=2, num_points_max=5):
        # Randomly select centroids from current cell
        num_init_centroids = randint(num_points_min, num_points_max)
        new_centroids = set()
        for centr_idx in range(num_init_centroids):
            centroid = []
            while True:
                for dim in range(VoronoiCell.input_dim):
                    centroid.append(random integer inside ridges) # TODO: see line 20 TODO (generalize this function to more than just initial cell (so not rectangular))
                centroid = tuple(centroid)
                if not centroid in new_centroids:
                    new_centroids.add(centroid)
                    break
        
        # Obtain Voronoi tessellation
        new_centroids = list(new_centroids)
        vor = Voronoi(new_centroids)
        bounded_ridges = [ridge for ridge in vor.ridges if not -1 in ridge]

        # Construct new cells from generated centroids and Voronoi tessellation
        new_cells = []
        for centr_idx in range(num_init_centroids):
            ridges = VoronoiCell.get_ridges(bounded_ridges, new_centroids[centr_idx], vor.vertices)
            new_cells.add(VoronoiCell(new_centroids[centr_idx], ridges))
        return new_cells
    
    @staticmethod
    def get_ridges_for_region(centroid, region, bounded_ridges, vertices): # Given a region and a set of bounded regions, return the ridges of this region as Hyperplane objects
        ridges_of_given_region = []
        for ridge in bounded_ridges:
            if set(ridge).issubset(region):
                ridges_of_given_region.append(VoronoiCell.get_hyperplane(centroid, ridge, vertices))
        return ridges_of_given_region

    @staticmethod
    def get_hyperplane(centroid, ridge, vertices): # Given a ridge, return it as a hyperplane
        vertices_array = np.array(vertices[ridge])

        # TODO: see line 20 TODO (find utility like np.linalg.lstsq for computing hyperplane from vertices; otherwise, review relevant linear algebra and find best approach)
        
        inequality_type = None
        if centroid dot coeffs < constant: 
            MarabouCore.Equation.EquationType.LE
        elif centroid dot coeffs < constant:
            MarabouCore.Equation.EquationType.GE
        else:
            raise ValueError("Centroid lies on VoronoiCell ridge")
    
        return Hyperplane(coeffs, constant, inequality_type)

def debug_log(*str):
    if DEBUG:
        print(str)

def form_query(network, output_var, curr_segment):
    for ridge in curr_segment.ridges:
        eq = MarabouCore.Equation(ridge.type)

        input_vars = network.input_vars[0][0]
        for var in input_vars:
            eq.addAddend(ridge.coeffs[var], var)
        eq.setconstant(ridge.constant)

        network.addEquation(eq)

    eq = MarabouCore.Equation(MarabouCore.Equation.EquationType)
    eq.addAddend(1, output_var)
    eq.setconstant(ridge.output)
    network.addEquation(eq)

def construct_initial_ridge_vertices(input_vars):
    init_ridges = []
    for var in input_vars:
        temp_coeffs = [0 if i != var else 1 for i in input_vars]
        init_ridges.append(Hyperplane(temp_coeffs, SYNTHESIS_UPPER_BOUND, MarabouCore.Equation.EquationType.LE))
        init_ridges.append(Hyperplane(temp_coeffs, SYNTHESIS_LOWER_BOUND, MarabouCore.Equation.EquationType.GE))
    return init_ridges

def main():
    network = Marabou.read_tf(filename = TF_NN_FILENAME, modelType="savedModel_v2")
    input_vars = network.input_vars[0][0]
    output_vars = network.output_vars[0].flatten()
    VoronoiCell.input_dim = len(input_vars)
    
    # Loop 1: Synthesize piecewise mapping for each output variable
    for i in tqdm(range(len(output_vars))):
        debug_log(f"--- Searching piecewise mapping for Output Variable {i} ---")
        output_var = output_vars[i]

        segments = set() # Set of centroids belonging to all VoronoiCell objects in input space
        incomplete_segments = [] # Stack of segments left to search for exceptions
        
        init_centroid = tuple([ (SYNTHESIS_LOWER_BOUND + SYNTHESIS_UPPER_BOUND) / 2 for i in range(len(input_vars)) ])
        initial_cell = VoronoiCell(init_centroid, construct_initial_ridge_vertices(input_vars), 0)
        incomplete_segments.extend(initial_cell.scatter_inference())

        while incomplete_segments:
            curr_segment = incomplete_segments.pop()

            network.clearProperty()
            form_query(network, output_var, curr_segment) # Use current segment inequalities on Marabou
            exitCode, vals, stats = network.solve(verbose=False) # Query Marabou for counterexamples within segment

            if len(vals):
                new_centroid = tuple([vals[var] for var in input_vars])
                # TODO: see line 20 (need to merge Voronoi tessellation of new centroid with non-uniform boundary of existing segment)
            else:
                segments.add(curr_segment)

if __name__ == '__main__':
    main()