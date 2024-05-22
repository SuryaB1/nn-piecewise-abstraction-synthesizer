# Project Directory Structure

This document provides an overview of the project directory structure to help new contributors understand the organization and purpose of each major directory and file.

## Documentation Directory
- **docs/**: Contains various documentation files.
  - **CONTRIBUTING.md**: Instructions for contributing to this project.
  - **info.md**: This file, describing the directory structure.
  - **todo.md**: List of issues that need to be resolved and next steps for the project.
  - **marabou_issues.md**: A log of all Issues submitted to the Marabou repository for this project.
  - **ridge_hyperplane_calculations.ipynb**: A Jupyter notebook listing methods to compute the equation of a hyperplane representing a Voronoi ridge in any dimension.
  - **scipy_voronoi.ipynb**: A Jupyter notebook to explain, demonstrate, and test the functionality of the `scipy.spatial.Voronoi` library.

## Source Code Directory
- **src/nn_piecewise_abstraction_synthesizer/**: Contains the source code of the project where `nn_piecewise_abstraction_synthesizer` is this project's package.
  - **__init__.py**: Initialization file for the Python package.
  - **main.py**: Main module for the NN piecewise abstraction synthesizer.
  - **utils/**: Utility modules.
    - **voronoi_cell.py**: Utility module containing the `VoronoiCell` class that has static functions for constructing a Voronoi tessellation and instances of which are cells of the current tessellation. It also defines the `Hyperplane` class to represent hyperplanes in Cartesian equations. This module utilizes `scipy.spatial.Voronoi` for tessellation and `maraboupy` for equation types and serves as the interface between Voronoi-based geometric calculations and Marabou's query encoding.
    - **voronoi_plot_2d.py**: Utility module for 2D Voronoi plotting. Contains the `voronoi_plot_2d_colored()` function, adapted from the `voronoi_plot_2d()` function in `scipy.spatial._plotutils`, to plot 2D Voronoi diagrams with coloring for Voronoi cells based on their associated output classes.

## Data Directory
- **data/inputs**: Contains various input data for use in the project.
  - **init_datapoints/**: Sets of sample datapoints to construct an initial Voronoi tessellation for CEGIS.
  - **models/**: Directory for TensorFlow model-related scripts and saved TensorFlow models.
    - **model_saving_scripts/**: Scripts used for saving TensorFlow models. To understand the model saved by each script in this directory, see the description of the corresponding saved model under the `saved_models/` bullet below.
    - **saved_models/**: Various saved TensorFlow models used in the project.
      - **2D_unit_sqr_classif_nn_using_sigmoid/**: A neural network that classifies whether or not a 2D point is on the first quadrant unit square. This network uses a sigmoid activation function rather than exclusively piecwise linear activation functions so it is not compatible with Marabou.
      - **2D_unit_sqr_classif_nn/**: Same as `2D_unit_sqr_classif_nn_using_sigmoid` with only piecewise linear activation functions.
      - **3D_unit_sqr_classif_nn/**: Same as `2D_unit_sqr_classif_nn` except with a unit cube rather than unit square.
      - **4D_unit_sqr_classif_nn/**: Same as `2D_unit_sqr_classif_nn` except with a unit tesseract rather than unit square.
      - **basic_nnet/**: A neural network representing the expression `3x + 2y`, where `x` and `y` are inputs.
      - **three-in_two-in_nnet/**: A neural network with three inputs and two outputs. Given x, y, and z are inputs, the first output is `3x + 2y`, and the second output is `z`.
      - **concave_polygon_classif_nn/**: A neural network that classifies whether or not a 2D point is on [this particular concave polygon](https://www.desmos.com/calculator/xu2oemndhd) formed by the intersection of five inequalities.
      - **convex_polygon_classif_nn/**: A neural network that classifies whether or not a 2D point is on [this particular convex polygon](https://www.desmos.com/calculator/ttlg7n5eun) formed by the intersection of four inequalities.
      - **disjoint_polygons_classif_nn/**: A neural network that classifies whether or not a 2D point is on either the concave polygon from `concave_polygon_classif_nn` or the convex polygon from `convex_polygon_classif_nn`. (**TODO**: see [todo.md](todo.md)).
      - **reluplex_paper_fig2_nn/**: A simple neural network from Figure 2 of [the ReLUPlex paper](https://arxiv.org/pdf/1702.01135).
      - **sign_classif_nn_using_softmax/**: A neural network with two outputs where one output represents the probability of the input value being a positive number and the other output represents the probability of the input value being a negative number. This network uses a softmax activation function rather than exclusively piecwise linear activation functions so it is not compatible with Marabou.
      - **sign_classif_nn/**: Same as `sign_classif_nn_using_softmax` with only piecewise linear activation functions, and, as a result, the outputs are, essentially, unnormalized probabilities.
      - **y=x_split_classif_nn/**: A neural network that distinguishes between points right of y=x and points on or to the left of y=x.
- **data/outputs**: Contains various output data generated by the project from various inputs.
  - **voronoi_cegis_output/**: Directory for Voronoi CEGIS GIFs. While files under `init_points_experiments/` are just PNGs showing the tessellated input space before CEGIS and after CEGIS, most sub-directories of this directory contains a GIF as well as PNGs representing the frames of the GIF corresponding to output under different hyperparameters, modifications of the implememtation, or different inputs. Each frame of the GIF is the Voronoi tessellation of the bounded input space at a particular iteration of the CEGIS loop where the specific iteration is denoted in the PNG filename.
    - **original_neg10_to_pos10/**: GIF output on 2D input space bounded in [-10, 10] on both axes.
    - **original_neg100_to_pos100/**: GIF output on 2D input space bounded in [-100, 100] on both axes.
    - **bfs_neg100_to_pos100**: GIF output on 2D input space bounded in [-100, 100] on both axes using a breadth-first search (BFS) approach for CEGIS. For context surrounding this experimentation of a BFS approach, whenever a segment is split in the current implementation, the segments are added to a stack, which is popped and queries during the next iteration of the CEGIS loop in a depth-first search (DFS) manner. The key takeaway from this experimentation with the BFS approach is that a greater number of queries are needed with the BFS approach than the DFS approach given arbitrary initial points in the input space.
    - **init_points_experiments/**: Contains outputs for experiments varying the initial datapoints for the Voronoi tessellation.
      - **random_init_points_few/**: Output given a few randomly generated initial datapoints.
      - **random_init_points_many/**: Output given many randomly generated initial datapoints.
      - **arranged_init_points/**: Output given initial datapoints datapoints arranged to create a tessellation that minimizes the effort (number of CEGIS loop iterations) needed in synthesis.
      
      **NOTE**: The overall takeaway from these experiments of varying the initial points is that fewer starting points are always preferred to more starting points in order to minimize the number of piecewise segments that need to be later merged, and spending effort to engineer the starting points in a way that minimizes the number of time and piecewise segments is important.

## Tests Directory
- **tests/**: Contains test scripts and directories.
  - **standalone_tests/**: Tests that can be run independently.
    - **test_individual_query.py**: A script to test an individual Marabou query.
    - **test_loaded_tf_nn.py**: A script to load and test a saved TensorFlow neural network.
  - **synthesizer_tests/**: Tests for the main Python package.

## Legacy Directory
- **legacy/**: Contains old implementations of this project.
  - **function_abstraction_cegis/**: Project directory for a synthesizer that synthesized abstractions of basic neural networks in the form of a function that represented the input-output relationship. This was the first version of this project implemented.
    - **main.py**: Implementation of function abstraction CEGIS.
    - **multiple_output_function_cegis.out**: Output file generated from inputting the `three_in_two_out_nn` saved model
    - **single_output_function_cegis.out**: Output file generated from inputting the `basic_nn` saved model
  - **rectangular_piecewise_abstraction_cegis/**: Project directory of a synthesizer that synthesized piecewise abstractions of classification neural networks where each segment is a rectangle (rather than a Voronoi cell). Here is a [conceptual walkthrough](https://docs.google.com/presentation/d/1qMwuZ1n9Nw8i9b3uD0gpT2coKixjOPPxM46aJkbJBpg/edit?usp=sharing) (a PDF version without animations is also in the `arbitrary_dim_input/` directory) of how segment splitting is implemented with rectangular segments (which occurs when there is a counterexample). This conceptual walkthrough covers the general case as well as edge cases when splitting rectangular segments, as denoted by the section headers in the conceptual walkthrough. Before a version of this synthesizer that worked for arbitrary-dimension inputs was developed (found in `multi_dim_input/`), a version that only worked on single-dimensional input was developed (found in `single_dim_input`) where each segment is a line segment rather than rectangle.
    - **arbitrary_dim_input/**: The latest version of this rectangular piecewise abstraction synthesizer. This version works for arbitrary-dimensional input.
      - **main.py**: Implementation.
      - **arbitrary_in_rect_piecewise.out**: Example output for the `2D_unit_sqr_classif_nn` saved model.
      - **tests/**: Test scripts that validate the output of this implementation. To use these test scripts, the `mappings` variable in the script being used needs to be set to the final output of implementation.
        - **test_unit_square_nn_case_2D_input.py**: This test validates the implementation's output given the unit square neural network test case with 2D input.
        - **test_unit_square_nn_case_3D_input.py**: This test validates the implementation's output given the unit square neural network test case with 3D input.
        - **test_unit_square_nn_case_4D_input.py**: This test validates the implementation's output given the unit square neural network test case with 4D input.
      - **rect_split_concept_walkthrough.pdf**: A conceptual walkthrough of how segment splitting is implemented with rectangular segments, which occurs when there is a counterexample. See the current directory's description above for a link to an animated version of the conceptual walkthrough.
    - **single_dim_input/**: The first version of this rectangular piecewise abstraction synthesizer. This version only works for single-dimensional input.
      - **main.py**: Implementation.
      - **single_in_rect_piecewise.out**: Example output.