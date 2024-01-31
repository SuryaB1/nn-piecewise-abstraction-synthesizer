from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
from random import randint
from dataclasses import dataclass
import ast

from maraboupy import Marabou
from maraboupy import MarabouCore
# from maraboupy import MarabouNetwork
# from maraboupy import MarabouUtils

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

MIN_COUNTEREX_DISTANCE_TO_RIDGE = 1 # minimum distance a counterexample can be from a segment boundary to be considered

@dataclass
class Hyperplane:
    coeffs: np.ndarray
    constant: float
    type: MarabouCore.Equation.EquationType

    def distance_to_hyperplane(self, point):
        unscaled_distance = np.dot(self.coeffs, point) + self.constant
        return unscaled_distance / np.linalg.norm(self.coeffs)

class VoronoiCell:
    vor = None # Voronoi tessellation object
    def __init__(self, centroid, ridges, output, neighbors):
        self.centroid = centroid
        self.ridges = ridges
        self.output = output
        self.neighbors = neighbors

    def compute_dist_to_closest_ridge(self, point):
        return min([ridge.distance_to_hyperplane(point) for ridge in self.ridges])
    
    @staticmethod
    def compute_voronoi_tessellation(input_output_dict, add_points=False):
        # Obtain updated Voronoi tessellation object
        new_centroids = list(input_output_dict.keys())
        first_new_centroid_idx = 0
        if add_points:
            first_new_centroid_idx = len(vor.points)
            vor.add_points(new_centroids)
        else:
            vor = Voronoi(new_centroids, incremental=True)

        # Construct new cells from generated centroids and Voronoi tessellation
        new_cells = []
        for i in range(first_new_centroid_idx, len(new_centroids)):
            centr = new_centroids[i]
            ridges, neighbors = VoronoiCell.get_ridges_and_neighbors_for_region(centr, i, vor.ridge_dict, vor.vertices)
            new_cells.append(VoronoiCell(centr, ridges, input_output_dict[centr], neighbors))

        # Update new cells' neighbors
        if add_points:
            curr_num_new_cells = len(new_cells)
            for c in range(curr_num_new_cells):
                for n in c.neighbors:
                    ridges, neighbors = VoronoiCell.get_ridges_and_neighbors_for_region(vor.points[n], n, vor.ridge_dict, vor.vertices)
                    new_cells.append(VoronoiCell(centr, ridges, input_output_dict[centr], neighbors))

        return new_cells
    
    @staticmethod
    def get_ridges_and_neighbors_for_region(centroid, region_idx, ridge_dict, vertices): # Given a region and a list of all ridges, return the ridges of this region as Hyperplane objects
        neighbors = []
        ridges = []

        # Centroid distribution characteristics
        center = vor.points.mean(axis=0)
        ptp_bound = vor.points.ptp(axis=0)

        # point_idx, simplex
        for adj_region_pair, ridge in ridge_dict.items():
            # NOTE: below will only work for 2D case (based on scipy.spatial.Voronoi.voronoi_plot_2d())
            region_pair_idx = adj_region_pair.index(region_idx)
            if region_pair_idx >= 0: # if region_idx is part of this pair of regions/centroids/segments
                ridge_hyperplane_vertices = vertices[ridge]
                for vert_idx in np.where(ridge == -1):
                    finite_end_ridge_vrtx = ridge[ridge >= 0][0]  # finite end of line segment ridge

                    tangent = vor.points[adj_region_pair[1]] - vor.points[adj_region_pair[0]]  # tangent
                    tangent /= np.linalg.norm(tangent)
                    normal = np.array([-tangent[1], tangent[0]])  # normal

                    midpoint = vor.points[adj_region_pair].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, normal)) * normal
                    far_point = vor.vertices[finite_end_ridge_vrtx] + direction * ptp_bound.max()

                    ridge_hyperplane_vertices[vert_idx] = far_point
                    
                ridges.append( VoronoiCell.get_hyperplane(centroid, ridge_hyperplane_vertices) )
                neighbors.append(adj_region_pair[ int(not region_pair_idx) ])

        return ridges, neighbors

    @staticmethod
    def get_hyperplane(centroid, hyperplane_vertices): # Given a ridge, return it as a hyperplane by solving Ax=b (where x is the coefficients) using least squares
        # Arbitrary-dimension case
        # A = np.array(vertices[ridge]) # Array of vertices
        # b = np.ones(A.shape[0])
        # coefficients = np.linalg.pinv(A) @ b # TODO: compare with np.linalg.lstsq for computing hyperplane from vertices
        # constant = 1
        
        # 2D case
        y_delta = hyperplane_vertices[1][1] - hyperplane_vertices[0][1]
        x_delta = hyperplane_vertices[1][0] - hyperplane_vertices[0][0]
        x_coeff = y_delta / x_delta if x_delta != 0 else -1
        y_coeff = int(x_delta != 0)
        constant = y_coeff * hyperplane_vertices[0][1] - x_coeff * hyperplane_vertices[0][0]
        coeffs = np.asarray([-1*x_coeff, y_coeff])

        if np.dot(coeffs, centroid) < constant: 
            inequality_type = MarabouCore.Equation.EquationType.LE
        elif np.dot(coeffs, centroid) > constant:
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
    network.clearProperty()

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
    output_var = output_vars[0]
    init_datapoints = read_init_datapoints("diagonal-split_classif_nnet_sample_datapoints.txt", len(input_vars), len(output_vars))
    
    # Note: This algorithm would be applied to a network with single output variable (for classification)
    centroid_cell_map = dict() # Dictionary of all centroids in input space mapped to corresponding VoronoiCell objects
    incomplete_segments = [] # Stack of centroids of segments left to search for counterexamples
    
    if init_datapoints:
        new_cells = VoronoiCell.compute_voronoi_tessellation(init_datapoints)
        for cell in new_cells:
            incomplete_segments.append(cell.centroid)
            centroid_cell_map[cell.centroid] = cell
    else: # if no user-given initial datapoint, set input space as the initial Voronoi cell
        init_centroid = tuple([ (SYNTHESIS_LOWER_BOUND + SYNTHESIS_UPPER_BOUND) / 2 for i in range(len(input_vars)) ])
        network.clearProperty()
        init_output = network.evaluateWithMarabou(init_centroid)
        initial_cell = VoronoiCell(init_centroid, construct_initial_ridge_vertices(input_vars), int(init_output), [])

        centroid_cell_map[init_centroid] = initial_cell
        incomplete_segments.append(initial_cell)

    while incomplete_segments:
        curr_segment = incomplete_segments.pop()

        form_query(network, output_var, centroid_cell_map[curr_segment]) # Use current segment inequalities on Marabou
        exitCode, vals, stats = network.solve(verbose=False) # Query Marabou for counterexamples within segment
        
        counterex_centroid = tuple([float(vals[var]) for var in input_vars])
        if len(vals) and centroid_cell_map[curr_segment].compute_dist_to_closest_ridge(counterex_centroid) > MIN_COUNTEREX_DISTANCE_TO_RIDGE:
            incomplete_segments.append(counterex_centroid)
            updated_cells = VoronoiCell.compute_voronoi_tessellation( {counterex_centroid : int(vals[output_var])}, add_points=True )
            for cell in updated_cells:
                centroid_cell_map[cell.centroid] = cell

if __name__ == '__main__':
    main()