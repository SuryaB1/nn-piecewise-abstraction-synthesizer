import sys
import random
import ast
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, _qhull
from dataclasses import dataclass

from maraboupy import Marabou
from maraboupy import MarabouCore
# from maraboupy import MarabouNetwork
from maraboupy import MarabouUtils
# TODO: look into how Voronoi is generating the ridges ince they don't seem perpendicular; hceck Marabou check implementaiton
# TODO: look into outer counter-examples step throuhg, clever way for merging
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
# SYNTHESIS_ORIGIN = (0, 0) # TODO: allow for custom input space origin

MIN_COUNTEREX_DISTANCE_TO_RIDGE = 0.5  # Minimum distance a counterexample can be from a segment boundary to be considered

centroid_cell_map = dict() # Dictionary of format {centroid in input space : corresponding VoronoiCell object}
                            # global because Voronoi.compute_voronoi_tessellation() uses it to determine updated neighbor output values (could have returned incorrect outputs for main function to correct with local centroid_cell_dict, but not clean)

@dataclass
class Hyperplane:
    """A class that represents a hyperplane as a Cartesian equation."""

    coeffs: np.ndarray
    constant: float
    type: MarabouCore.Equation.EquationType

    def __str__(self):
        return f''

    def __repr__(self):
        return f'Hyperplane(\'{self.coeffs}\', {self.constant}\', {self.type})'

    # def distance_to_hyperplane(self, point):
    #     unscaled_distance = np.dot(self.coeffs, point) + self.constant
    #     return unscaled_distance / np.linalg.norm(self.coeffs)

class VoronoiCell:
    """A class that represents a Voronoi tessellation with instances representing individual cells."""
    
    vor = None  # Voronoi tessellation object
    def __init__(self, centroid, ridges, output, neighbors):
        self.centroid = centroid # Tuple of coordinate component floats
        self.ridges = ridges  # List of Hyerplane objects
        self.output = output  # Int
        self.neighbors = neighbors  # List of centroid indices

    def __str__(self):
        return f''

    def __repr__(self):
        return f'VoronoiCell(\'{self.centroid}\', {self.ridges}\', {self.output}\', {self.neighbors})'

    # def compute_dist_to_closest_ridge(self, point):
    #     return min([ridge.distance_to_hyperplane(point) for ridge in self.ridges])
    
    @staticmethod
    def compute_voronoi_tessellation(input_output_dict, add_points=False):
        # Obtain updated Voronoi tessellation object
        new_centroids = list(input_output_dict.keys())
        first_new_centroid_idx = 0
        if add_points:
            first_new_centroid_idx = len(VoronoiCell.vor.points)
            VoronoiCell.vor.add_points(new_centroids)
        else:
            VoronoiCell.vor = Voronoi(new_centroids, incremental=True)

        # Construct new cells from generated centroids and Voronoi tessellation
        new_cells = []
        new_cells_centr_idxs = set()
        # print(f"{first_new_centroid_idx}, {len(new_centroids)}")
        for i in range(first_new_centroid_idx, len(VoronoiCell.vor.points)):
            centr = tuple(VoronoiCell.vor.points[i])
            print(f"centr: {centr}")
            ridges, neighbors = VoronoiCell.get_ridges_and_neighbors_for_region(centr, i, VoronoiCell.vor.ridge_dict, VoronoiCell.vor.vertices)
            new_cells.append(VoronoiCell(centr, ridges, input_output_dict[centr], neighbors))
            new_cells_centr_idxs.add(i)

        # Update new cells' neighbors (if the neighbor is not one of the new cells)
        if add_points:
            for c in range(len(new_cells)):
                for n in new_cells[c].neighbors:
                    if not n in new_cells_centr_idxs:
                        centr = tuple(VoronoiCell.vor.points[n])
                        ridges, neighbors = VoronoiCell.get_ridges_and_neighbors_for_region(centr, n, VoronoiCell.vor.ridge_dict, VoronoiCell.vor.vertices)
                        new_cells.append(VoronoiCell(centr, ridges, centroid_cell_map[centr].output, neighbors))

        return new_cells
    
    @staticmethod
    def get_ridges_and_neighbors_for_region(centroid, region_idx, ridge_dict, vertices):
        """Given a region and all of its ridges, this function will return the ridges of this region as Hyperplane objects and neighbors of this region"""
        neighbors = []
        ridges = []

        # Centroid distribution characteristics
        center = VoronoiCell.vor.points.mean(axis=0)
        ptp_bound = VoronoiCell.vor.points.ptp(axis=0)

        # Loop through each region index pair (formatted as (region index, adjacent region index)) and the ridge in between
        for adj_region_pair, ridge in ridge_dict.items():
            # NOTE: below code will only work for 2D case (based on scipy.spatial.Voronoi.voronoi_plot_2d(), and ne)
            ridge = np.array(ridge)

            if region_idx in adj_region_pair:
                region_pair_idx = adj_region_pair.index(region_idx)

                # If region_idx is part of this pair of regions (equivalent to centroids and segments), extract ridge and neighbor data
                ridge_hyperplane_vertices = vertices[ridge]

                # Approximate every infinite vertex of the common ridge of this adjacent region apir
                for vert_idx in np.where(ridge == -1):  # Single iteration in 2D case
                    finite_end_ridge_vertex = ridge[ridge >= 0][0]  # Finite end of line segment ridge in 2D case

                    tangent = VoronoiCell.vor.points[adj_region_pair[1]] - VoronoiCell.vor.points[adj_region_pair[0]]  # Tangent of adjacent centroids
                    tangent /= np.linalg.norm(tangent)
                    normal = np.array([-tangent[1], tangent[0]])  # Normal vector to the tangent line

                    midpoint = VoronoiCell.vor.points[list(adj_region_pair)].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, normal)) * normal
                    far_point = VoronoiCell.vor.vertices[finite_end_ridge_vertex] + direction * ptp_bound.max()  # Approximation of the infinite vertex

                    ridge_hyperplane_vertices[vert_idx] = far_point  # Save the approximation into the index of the infinite vertex in the corresponding ridge's vertices
                    
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
        assert not (x_delta == 0 and y_delta == 0)

        x_coeff = y_delta / x_delta if x_delta != 0 else -1.0
        y_coeff = float(x_delta != 0)
        coeffs = np.asarray([-1*x_coeff, y_coeff])

        constant = np.dot(coeffs, hyperplane_vertices[0])
        
        assert np.dot(coeffs, centroid) != constant  # Centroid should not lie on VoronoiCell ridge; if it does, its an implementation issue
        if np.dot(coeffs, centroid) < constant: 
            inequality_type = MarabouCore.Equation.LE
        elif np.dot(coeffs, centroid) > constant:
            inequality_type = MarabouCore.Equation.GE
    
        return Hyperplane(coeffs, constant, inequality_type)

def debug_log(*str):
    if DEBUG:
        print("DEBUG:", *str)

# Assumes input data is flattened to single dimensional Python list, and output class is integer
def read_init_datapoints(filename="", input_dim=0):
    if filename == "": return dict()

    input_output_dict = dict()
    file_type = filename.split(".")[-1]
    if file_type == "json":
        pass  # TODO
    elif file_type == "txt":
        file = open(filename, 'r')
        lines = file.readlines()

        # Parse input file lines, and populate input_output_dict
        for line in lines:
            split_line = line.split(':')
            
            input_coord = tuple([float(element) for element in ast.literal_eval(split_line[0].strip())])
            assert len(input_coord) == input_dim

            output = ast.literal_eval(split_line[1].strip())

            input_output_dict[input_coord] = output
    return input_output_dict

def form_query(network, output_var, curr_segment):
    network.clearProperty()

    # Add inequalities for synthesis input space bounds to query
    for var in network.inputVars[0][0]:
        network.setLowerBound(var, SYNTHESIS_LOWER_BOUND)
        network.setUpperBound(var, SYNTHESIS_UPPER_BOUND)

    # Add inequalities for current segment's ridges
    for ridge in curr_segment.ridges:
        eq = MarabouUtils.Equation(ridge.type)

        input_vars = network.inputVars[0][0]
        for var in input_vars: # for var_idx in range(0, input_vars): var = input_vars[var_idx] eq.addAddend(ridge.coeffs[var_idx], var)
            eq.addAddend(ridge.coeffs[var], var)
        eq.setScalar(ridge.constant)

        network.addEquation(eq, isProperty=True)

    # Add equality expressing class assignment for current segment
    eq = MarabouUtils.Equation()
    eq.addAddend(1, output_var)
    eq.setScalar(int(not curr_segment.output)) # TODO: works only for binary classification case, need to change to LE constant - 1, and GE constant + 1 for multiple classes
    network.addEquation(eq, isProperty=True)

def main():
    # NOTE #1: Run program with Python version X.Y where X and Y are defined in MarabouCore.cpython-XY-darwin.so 
    #   located in the separate directory Marabou/maraboupy/, which can be obtained during maraboupy build process

    # NOTE #2: MarabouUtils.Equation() may be deprecated and replaced with MarabouCore.Equation() in the near future; 
    #   at the moment, MarabouUtils.Equation() is what works with MarabouNetwork.addEquation(), and, thus, is used in this program

    # Read single-output classification neural network into Marabou
    network = Marabou.read_tf(filename = TF_NN_FILENAME, modelType="savedModel_v2")  # Returns MarabouNetworkTF object
    input_vars = network.inputVars[0][0]
    output_vars = network.outputVars[0].flatten()
    assert len(output_vars) == 1  # Ensure that network has a single output
    output_var = output_vars[0]
    # SYNTHESIS_ORIGIN = tuple([0 for i in range(len(input_vars))])
    
    debug_log("Read input neural network. Parsing input datapoints...")

    # Parse datapoints provided for initialization of Voronoi tessellation
    init_datapoints = read_init_datapoints("sample_data/diagonal-split-sample-data.txt", len(input_vars))  # diagonal-split-sample-data
    num_init_datapoints = len(init_datapoints)
    min_num_init_datapoints = 4 + (len(input_vars) - 2)  # Minimum number of datapoints needed to compute Voronoi tessellation by scipy.spatial.Voronoi 
    if num_init_datapoints > 0 and num_init_datapoints < min_num_init_datapoints:
        sys.exit(f"ERROR: At least {min_num_init_datapoints} input datapoints are needed, but only {len(init_datapoints)} were given.")
    
    # Initialize data structures for keeping track of segments
    incomplete_segments = [] # Stack of centroids representing unsearched segments
    
    # If no initial datapoints are provided, generate initial datapoints in input space
    if not init_datapoints:
        # Select centroids representing initial datapoints in input space (currently, relative to origin - see line 26 TODO)
        init_centroids = itertools.product( *[ [-0.5*SYNTHESIS_LOWER_BOUND, 0.5*SYNTHESIS_UPPER_BOUND] for i in range(len(input_vars)) ] )
        
        # Convert centroid to datapoint (since scipy.spatial.Voronoi is not compatible with points being co-circular/co-spherical)
        for coord in init_centroids:
            # Add noise to centroid
            random_idx_for_noise = random.randint(0, len(input_vars) - 1)  # Select a dimension of the centroid to receive noise
            random_noise_offset = random.uniform(-1, 1)  # Compute noise to apply to centroid
            coord[random_idx_for_noise] += random_noise_offset

            # Compute output value of centroid
            network.clearProperty()
            init_output = network.evaluateWithMarabou(coord)
            init_datapoints[tuple(coord)] = init_output

    # Compute initial Voronoi Tessellation with initial datapoints
    try:
        new_cells = VoronoiCell.compute_voronoi_tessellation(init_datapoints)
    except _qhull.QhullError as e:  # Provide clear error message if datapoints are co-circular or co-spherical
        if "initial Delaunay input sites are cocircular or cospherical" in e:
            print("NOTE: This error message indicates that your datapoints are perfectly spaced out, which is incompatible with scipy.spatial.Voronoi incremental mode. Please add some slight noise to your datapoints.")
        raise

    # Populate data structures
    for cell in new_cells:
        incomplete_segments.append(cell.centroid)
        centroid_cell_map[cell.centroid] = cell

    debug_log("Populated data structures. Beginning CEGIS loop...")

    plt.rcParams["figure.figsize"] = (7, 7)
    voronoi_plot_2d(VoronoiCell.vor)
    class_boundary_x = np.linspace(SYNTHESIS_LOWER_BOUND*1000, SYNTHESIS_UPPER_BOUND*1000, 100)
    class_boundary_y = class_boundary_x
    plt.plot(class_boundary_x, class_boundary_y)

    # CEGIS loop
    while incomplete_segments:
        curr_segment = incomplete_segments.pop()
        debug_log("Current segment:", curr_segment)

        # Query Marabou for counterexample within segment
        form_query(network, output_var, centroid_cell_map[curr_segment])
        exitCode, vals, stats = network.solve(filename="marabou.log", verbose=False)

        # Split current segment if counterexample exists (not doing this anymore since Marabou has its own min precision: and counterexample is not too close to segment boundary)
        if len(vals):
            counterex_centroid = tuple([float(vals[var]) for var in input_vars])  # Convert Marabou counterexample format to coordinate
            # if centroid_cell_map[curr_segment].compute_dist_to_closest_ridge(counterex_centroid) > MIN_COUNTEREX_DISTANCE_TO_RIDGE:
            debug_log("Splitting segment", curr_segment, "with counterexample", counterex_centroid)
            incomplete_segments.append(counterex_centroid)

            # Update Voronoi tessellation based on counterexample
            updated_cells = VoronoiCell.compute_voronoi_tessellation( {counterex_centroid : int(vals[output_var])}, add_points=True )
            for cell in updated_cells:
                centroid_cell_map[cell.centroid] = cell


    voronoi_plot_2d(VoronoiCell.vor)
    class_boundary_x = np.linspace(SYNTHESIS_LOWER_BOUND*1000, SYNTHESIS_UPPER_BOUND*1000, 100)
    class_boundary_y = class_boundary_x
    plt.plot(class_boundary_x, class_boundary_y)
    plt.show()

if __name__ == '__main__':
    main()