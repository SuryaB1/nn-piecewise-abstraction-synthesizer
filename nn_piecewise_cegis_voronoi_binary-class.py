import sys
import os
import glob
import random
import ast
import itertools
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, _qhull
from voronoi_plot_2d_custom import voronoi_plot_2d_colored
from voronoi_cegis_utils import *

from maraboupy import Marabou
from maraboupy import MarabouCore
# from maraboupy import MarabouNetwork
from maraboupy import MarabouUtils

# Test cases
# TF_NN_FILENAME = "saved_models/sign_classif_nn_no-softmax" # 1D input
# TF_NN_FILENAME = "saved_models/unit-sqr_classif_nnet_no-sigmoid" # 2D input
TF_NN_FILENAME = "saved_models/diagonal-split_classif_nnet" # 2D input, non-rectangular case
# TF_NN_FILENAME = "saved_models/concave-poly_classif_nnet" # 2D input, non-rectangular case
# TF_NN_FILENAME = "saved_models/3d-unit-sqr_classif_nnet_no-sigmoid" # 3D input
# TF_NN_FILENAME = "saved_models/4d-unit-sqr_classif_nnet_no-sigmoid" # 4D input

DEBUG = True
OUTPUT_TESSELLATION_FORMATION_GIF = False

# Bounds on all axes of input space for which piecewise mapping is synthesized
SYNTHESIS_LOWER_BOUND = -10
SYNTHESIS_UPPER_BOUND = 10
# SYNTHESIS_ORIGIN = (0, 0) # TODO: allow for custom input space origin

MIN_COUNTEREX_DISTANCE_TO_RIDGE = 1  # Minimum distance a counterexample can be from a segment boundary to be considered (around 0.5 seems to be default from Marabou without using this param)
MIN_COUNTEREX_DIST_TO_CENTR = 0.01 # min for lib: 0.0001

def clear_directory(directory="tess_form_gif"):
    files = glob.glob(os.path.join(directory, '*'))
    for f in files:
        os.remove(f)

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

    omit_close_counterex_disjunction = []  # Inequalities to prevent counterexamples to close to curr_segment's centroid

    # Add inequalities for synthesis input space bounds to query
    for var in network.inputVars[0][0]:
        network.setLowerBound(var, SYNTHESIS_LOWER_BOUND)
        network.setUpperBound(var, SYNTHESIS_UPPER_BOUND)

        # Add inequalities to prevent counter-example from being too close current segment's centroid
        eq1 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        eq1.addAddend(1.0, var)
        eq1.setScalar(curr_segment.centroid[var] - MIN_COUNTEREX_DIST_TO_CENTR)

        eq2 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        eq2.addAddend(1.0, var)
        eq2.setScalar(curr_segment.centroid[var] + MIN_COUNTEREX_DIST_TO_CENTR)

        omit_close_counterex_disjunction.extend([[eq1], [eq2]])

    network.addDisjunctionConstraint(omit_close_counterex_disjunction)

    # Add inequalities for current segment's ridges
    for ridge in curr_segment.ridges:
        eq = MarabouUtils.Equation(ridge.type)

        input_vars = network.inputVars[0][0]
        for var in input_vars:
            eq.addAddend(ridge.coeffs[var], var)
        eq.setScalar(ridge.constant)

        network.addEquation(eq, isProperty=True)  # Need to isProperty set to true to be able to clear it with network.clearProperty()
    
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
    centroid_cell_map = dict() # Dictionary of format {centroid in input space : corresponding VoronoiCell object}
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
        raise e

    # Populate data structures
    for cell in new_cells:
        incomplete_segments.append(cell.centroid)
        centroid_cell_map[cell.centroid] = cell

    debug_log("Populated data structures. Beginning CEGIS loop...")

    plt.rcParams["figure.figsize"] = (7, 7)
    voronoi_plot_2d_colored(VoronoiCell.vor, centroid_cell_map=centroid_cell_map)
    class_boundary_x = np.linspace(SYNTHESIS_LOWER_BOUND*1000, SYNTHESIS_UPPER_BOUND*1000, 100)
    class_boundary_y = class_boundary_x
    plt.plot(class_boundary_x, class_boundary_y, color='g')

    if OUTPUT_TESSELLATION_FORMATION_GIF:
        tess_form_gif_fnames = []
        clear_directory()
        cegis_iteration = 0
        output_plot_filename = f'tess_form_gif/cegis_iteration_{cegis_iteration}.png'
        tess_form_gif_fnames.append(output_plot_filename)
        plt.savefig(output_plot_filename)

    # CEGIS loop
    while incomplete_segments: # TODO: set to just see front instead of pop to be more efficient
        curr_segment = incomplete_segments.pop()
        debug_log("Current segment:", curr_segment)

        # Query Marabou for counterexample within segment
        form_query(network, output_var, centroid_cell_map[curr_segment])
        options = Marabou.createOptions(verbosity = 0)
        exitCode, vals, stats = network.solve(options = options, verbose=False)

        # Split current segment if counterexample exists (not doing this anymore since Marabou has its own min precision: and counterexample is not too close to segment boundary)
        if len(vals):
            counterex_centroid = tuple([float(vals[var]) for var in input_vars])  # Convert Marabou counterexample format to coordinate
            # if centroid_cell_map[curr_segment].compute_dist_to_farthest_ridge(counterex_centroid) > MIN_COUNTEREX_DISTANCE_TO_RIDGE:
            debug_log("Splitting segment", curr_segment, "with counterexample", counterex_centroid)
            incomplete_segments.append(curr_segment)
            incomplete_segments.append(counterex_centroid)
            
            # Update Voronoi tessellation with counterexample
            updated_cells = VoronoiCell.compute_voronoi_tessellation({counterex_centroid : int(vals[output_var])}, add_points=True, centroid_cell_map=centroid_cell_map)
            for cell in updated_cells:
                centroid_cell_map[cell.centroid] = cell

            if OUTPUT_TESSELLATION_FORMATION_GIF:
                voronoi_plot_2d_colored(VoronoiCell.vor, centroid_cell_map=centroid_cell_map)
                class_boundary_x = np.linspace(SYNTHESIS_LOWER_BOUND*1000, SYNTHESIS_UPPER_BOUND*1000, 10)
                class_boundary_y = class_boundary_x
                plt.plot(class_boundary_x, class_boundary_y, color='g')

                cegis_iteration += 1
                output_plot_filename = f'tess_form_gif/cegis_iteration_{cegis_iteration}.png'
                tess_form_gif_fnames.append(output_plot_filename)
                plt.savefig(output_plot_filename)

    voronoi_plot_2d_colored(VoronoiCell.vor, centroid_cell_map=centroid_cell_map)
    class_boundary_x = np.linspace(SYNTHESIS_LOWER_BOUND*1000, SYNTHESIS_UPPER_BOUND*1000, 10)
    class_boundary_y = class_boundary_x
    plt.plot(class_boundary_x, class_boundary_y, color='g')

    if OUTPUT_TESSELLATION_FORMATION_GIF:
        image_np_arrays = [imageio.imread(fname) for fname in tess_form_gif_fnames]
        imageio.mimsave('tess_form_gif/tessellation_formation.gif', image_np_arrays)
    else:
        plt.show()

if __name__ == '__main__':
    main()