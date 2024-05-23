import sys
import os
import glob
import random
import ast
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, _qhull
from utils.voronoi_plot_2d import voronoi_plot_2d_colored
from utils.voronoi_cell import *

from maraboupy import Marabou
from maraboupy import MarabouCore
# from maraboupy import MarabouNetwork
from maraboupy import MarabouUtils

# Test cases
# TF_NN_FILENAME = "data/inputs/models/saved_models/sign_classif_nn" # 1D input
# TF_NN_FILENAME = "data/inputs/models/saved_models/2D_unit_sqr_classif_nn" # 2D input
TF_NN_FILENAME = "data/inputs/models/saved_models/y=x_split_classif_nn" # 2D input, non-rectangular class boundary test case
# TF_NN_FILENAME = "data/inputs/models/saved_models/concave_polygon_classif_nn" # 2D input, non-rectangular class boundary test case
# TF_NN_FILENAME = "data/inputs/models/saved_models/3D-unit-sqr_classif_nn" # 3D input
# TF_NN_FILENAME = "data/inputs/models/saved_models/4D-unit-sqr_classif_nn" # 4D input

DEBUG = True  # Set to true to print all debug output printed using debug_log("...", ...)
OUTPUT_TESSELLATION_FORMATION_GIF = True  # Set to true to output a GIF of the CEGIS process
STARTING_POINTS_PROVIDED = True  # Set to true if providing sample datapoints for CEGIS to start from

# Bounds on all axes of input space for which piecewise mapping is synthesized
SYNTHESIS_LOWER_BOUND = -10
SYNTHESIS_UPPER_BOUND = 10

MIN_COUNTEREX_DIST_TO_CENTR = 0.01  # Minimum distance a newly found counterexample can be from an existing centroid (must be ≥ 0.0001, since otherwise scipy.spatial.Voronoi will ignore this centroid causing further issues)

def clear_directory(directory="tess_form_gif"):
    files = glob.glob(os.path.join(directory, '*'))
    for f in files:
        os.remove(f)

def debug_log(*str):
    if DEBUG:
        print("[DEBUG]", *str)

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

def plot_piecewise_mappings(centroid_cell_map):
    voronoi_plot_2d_colored(VoronoiCell.vor, centroid_cell_map=centroid_cell_map)
    class_boundary_x = np.linspace(SYNTHESIS_LOWER_BOUND*100, SYNTHESIS_UPPER_BOUND*100, 10)
    class_boundary_y = class_boundary_x
    plt.plot(class_boundary_x, class_boundary_y, color='g')

def form_query(network, output_var, curr_segment):
    debug_log("PRINTING QUERY...")
    network.clearProperty()

    omit_close_counterex_disjunction = []  # Inequalities to prevent counterexamples to close to curr_segment's centroid

    debug_log("Disjunctions:")
    # Add inequalities for synthesis input space bounds to query
    for var in network.inputVars[0][0]:
        network.setLowerBound(var, SYNTHESIS_LOWER_BOUND)
        network.setUpperBound(var, SYNTHESIS_UPPER_BOUND)

        # Add inequalities to prevent counter-example from being too close current segment's centroid
        eq1 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        eq1.addAddend(1.0, var)
        eq1.setScalar(curr_segment.centroid[var] - MIN_COUNTEREX_DIST_TO_CENTR)
        debug_log(f"x{var} <= {curr_segment.centroid[var] - MIN_COUNTEREX_DIST_TO_CENTR}")

        eq2 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        eq2.addAddend(1.0, var)
        eq2.setScalar(curr_segment.centroid[var] + MIN_COUNTEREX_DIST_TO_CENTR)
        debug_log(f"x{var} >= {curr_segment.centroid[var] + MIN_COUNTEREX_DIST_TO_CENTR}")
        
        omit_close_counterex_disjunction.extend([[eq1], [eq2]])

    network.addDisjunctionConstraint(omit_close_counterex_disjunction)

    debug_log("Inequalities:")
    # Add inequalities for current segment's ridges
    for ridge in curr_segment.ridges:
        debug_log(ridge)
        eq = MarabouUtils.Equation(ridge.type)

        input_vars = network.inputVars[0][0]
        for var in input_vars:
            eq.addAddend(ridge.coeffs[var], var)
        eq.setScalar(ridge.constant)

        network.addEquation(eq, isProperty=True)  # Need to set isProperty to true to be able to clear this property with network.clearProperty()
    
    # Add equality expressing class assignment for current segment
    eq = MarabouUtils.Equation()
    eq.addAddend(1, output_var)
    eq.setScalar(int(not curr_segment.output)) # TODO: Scale implementation to handle more than two output classes (see todo.md), need to change to LE constant - 1, and GE constant + 1 for multiple classes
    network.addEquation(eq, isProperty=True)
    print(f"output_var=={int(not curr_segment.output)}")

def main():
    # Read single-output classification neural network into Marabou
    network = Marabou.read_tf(filename = TF_NN_FILENAME, modelType="savedModel_v2")  # Returns MarabouNetworkTF object
    input_vars = network.inputVars[0][0]
    output_vars = network.outputVars[0].flatten()
    assert len(output_vars) == 1  # Ensure that network has a single output
    output_var = output_vars[0]
    
    debug_log("Read input neural network. Parsing input datapoints...")

    # Parse datapoints provided for initialization of Voronoi tessellation
    init_datapoints = dict()
    min_num_init_datapoints = 4 + (len(input_vars) - 2)  # Minimum number of datapoints needed to compute Voronoi tessellation by scipy.spatial.Voronoi 
    if (STARTING_POINTS_PROVIDED):
        sample_data_filename = "y=x_split_arranged_data.txt" #  "y=x_split_sample_data.txt" # 
        path_to_init_datapoints = "data/inputs/init_datapoints/"
        init_datapoints = read_init_datapoints(f"{path_to_init_datapoints}{sample_data_filename}", len(input_vars))
        num_init_datapoints = len(init_datapoints)
        if num_init_datapoints > 0 and num_init_datapoints < min_num_init_datapoints:
            sys.exit(f"ERROR: At least {min_num_init_datapoints} input datapoints are needed, but only {len(init_datapoints)} were given.")
    else:  # If no initial datapoints are provided, sample initial datapoints in input space
        for _ in range(min_num_init_datapoints):
            init_centroid = tuple([ round(random.uniform(SYNTHESIS_LOWER_BOUND, SYNTHESIS_UPPER_BOUND), 2) for d in range( len(input_vars) ) ])
            while init_centroid in init_datapoints:  # Re-generate sample until unique sample found
                init_centroid = tuple([ round(random.uniform(SYNTHESIS_LOWER_BOUND, SYNTHESIS_UPPER_BOUND), 2) for d in range( len(input_vars) ) ])

            # Compute output value of centroid
            network.clearProperty()
            init_output = int(network.evaluateWithMarabou(init_centroid)[0][0])
            init_datapoints[init_centroid] = init_output
    
    debug_log(init_datapoints)
    
    # Compute initial Voronoi Tessellation with initial datapoints
    try:
        new_cells = VoronoiCell.compute_voronoi_tessellation(init_datapoints)
    except _qhull.QhullError as e:  # Provide clear error message if datapoints are co-circular or co-spherical
        if "initial Delaunay input sites are cocircular or cospherical" in e:
            print("NOTE: This error message indicates that your datapoints are perfectly spaced out, which is incompatible with scipy.spatial.Voronoi incremental mode. Please add some slight noise to your datapoints.")
        raise e
    
    # Initialize data structures for keeping track of segments
    centroid_cell_map = dict() # Dictionary of format {centroid in input space : corresponding VoronoiCell object}
    incomplete_segments = [] # Stack of centroids representing unsearched segments

    # Populate data structures
    for cell in new_cells:
        incomplete_segments.append(cell.centroid)
        centroid_cell_map[cell.centroid] = cell

    plt.rcParams["figure.figsize"] = (7, 7)
    plot_piecewise_mappings(centroid_cell_map)  # Plot initial Voronoi tessellation

    # Set up GIF output if applicable
    if OUTPUT_TESSELLATION_FORMATION_GIF:
        dirname = "tess_form_gif"
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        tess_form_gif_fnames = []
        clear_directory()
        cegis_iteration = 0
        output_plot_filename = f'tess_form_gif/cegis_iteration_{cegis_iteration}.png'
        tess_form_gif_fnames.append(output_plot_filename)
        plt.savefig(output_plot_filename)

    debug_log("Initialized data structures. Beginning CEGIS loop...")

    # CEGIS loop
    while incomplete_segments:
        curr_segment = incomplete_segments.pop()
        debug_log("Current segment:", curr_segment)

        # Query Marabou for counterexample within segment 
        # (once query times out, attempt query again to get lucky 
        #   with random branching order; see Issue #783 in docs/marabou_issues.md)
        exitCode = "TIMEOUT"
        query_timeout = 2  # seconds
        while exitCode[0] == "T": 
            if query_timeout > 2:
                debug_log(f"QUERY TIMED OUT. Requerying with timeut of {query_timeout} seconds...")
            debug_log("FORMING QUERY")
            form_query(network, output_var, centroid_cell_map[curr_segment])
            options = Marabou.createOptions(verbosity = 0, timeoutInSeconds=query_timeout)
            debug_log("QUERYING")
            exitCode, vals, stats = network.solve(options = options, verbose=False)
            query_timeout *= 2
        debug_log("QUERIED")

        # Split current segment if counterexample exists 
        if len(vals):
            counterex_centroid = tuple([float(vals[var]) for var in input_vars])  # Convert Marabou counterexample format to coordinate

            debug_log("Splitting segment", curr_segment, "with counterexample", counterex_centroid)
            incomplete_segments.append(curr_segment)
            incomplete_segments.append(counterex_centroid)
            
            # Update Voronoi tessellation with counterexample
            updated_cells = VoronoiCell.compute_voronoi_tessellation({counterex_centroid : int(vals[output_var])}, add_points=True, centroid_cell_map=centroid_cell_map)
            for cell in updated_cells:
                centroid_cell_map[cell.centroid] = cell

            if OUTPUT_TESSELLATION_FORMATION_GIF:
                plot_piecewise_mappings(centroid_cell_map)

                cegis_iteration += 1
                output_plot_filename = f'tess_form_gif/cegis_iteration_{cegis_iteration}.png'
                tess_form_gif_fnames.append(output_plot_filename)
                plt.savefig(output_plot_filename)

    plot_piecewise_mappings(centroid_cell_map)
    
    if OUTPUT_TESSELLATION_FORMATION_GIF:
        image_np_arrays = [imageio.imread(fname) for fname in tess_form_gif_fnames]
        imageio.mimsave('tess_form_gif/tessellation_formation.gif', image_np_arrays)
    else:
        plt.show()

if __name__ == '__main__':
    main()