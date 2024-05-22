# TODO List

## Bugs
- **Fix inaccurate hyperplane offset**

  In `get_hyperplane()`, which is in the `voronoi_cells.py` utilities module, 
  the hyperplane equation of the ridge input is offsetted by an `epsilon` amount 
  in the direction of the centroid to approximate a strict inequality for the Marabou 
  query. The current implementation for this offsetting, which is marked by the
  second TODO comment in the function, always offsets the hyperplane by an amount that is slightly 
  greater than epsilon, rather than exactly epsilon. This issue, which lies in the 
  linear algebra being performed, needs to be fixed.

- **Fix `disjoint_polygons_classif_nn` [model-saving script](../data/inputs/models/model_saving_scripts/output_disjoint_polygons_classif_nn.py)**

  Need to fix the model architecture to perform the union between the convex and concave polygons, rather than intersection as it is currently doing.

## Features
- **Implement functionality to merge cells during CEGIS**
  
  Currently, the implementation populates parts of the input space that correspond to the same output class with numerous Voronoi cells (see output GIFs in `data/outputs/voronoi_cegis_output_gifs`). This behavior (multiple adjacent cells mapping to the same output class) occurs due to the nature of the CEGIS, specifically, as new counterexamples are found and larger cells are split into smaller and smaller cells, especially closer to class boundaries.

  The downside to this behavior though, are that the number of cells would grow rapidly relative to the number of dimensions in the input space. Another downside is that, since much smaller cells are found across the class boundaries, the synthesized class boundary estimate zigzags to a significant degree around the true class boundary (see final plots in `data/outputs/voronoi_cegis_output_gifs`).
  
  One way to address this is:
  1. Select a cell along the class boundary (in other words, a cell that has a neighbor mapping to a different class)
  2. Store the ridge between the cell and the opposite-class neighbor; in other words, a ridge along the true class boundary
  3. Iterate though adjacent neighbor that are of the same class and are also along the class boundary
  4. As you iterate through the adjacent neighbors, compute an "average ridge" of all the ridges that have been along the class boundary
  5. If the average ridge ever deviates from the previous value of the average ridge by a certain threshold, save the average ridge as the new ridge for all the cells that have just been traversed, and restart the construction of a ridge starting from the current cell.

  This will reduce the occurence of zigzagging by approximating the class boundary _after the CEGIS process completes_ based on the synthesized cells.
  
  A potentially **more efficient** solution would be to merge every new cell (that result from the splitting of larger cells) with an adjacent cells of the same output class. Such a solution needs to be implemented.

- **Scale implementation to handle arbitrary-dimensional input**
  
  Currently, the implementation only accepts and generates a piecewise abstraction for 
  classification neural networks with 2D inputs. The main adjustments that need to be 
  made for this feature are concentrated in three parts of the implementation:

  1. In `get_hyperplane()`, which is in the `voronoi_cell.py` utilities module, the code marked by the first TODO comment of the function, which is responsible for computing a hyperplane equation that represents the given ridge, is currently only implemented for two-dimensional ridges. This code needs to be re-written to be able to compute the hyperplane equation for a ridge of any dimension. Some potential solutions are discussed in `ridge_hyperplane_calculations.ipynb`, but further research may be needed.

  2. In `get_ridges_and_neighbors_for_region()`, which is in the `voronoi_cell.py` utilities module, the code marked by the TODO comment, which is responsible for approximating an end-point for infinitely-extending 2D ridges, needs to be re-written to approximate endings for infinitely-extending ridges of arbitrary dimension.

- **Scale implementation to handle multiple output classes**
  
  Some parts of the current implementation (e.g. the TODO comment in `form_query()`) function under only two output classes being possible; this needs to be adjusted to allow for an arbitrary number of output classes.

## Enhancements
- **Improve visualization**
  
  Expand `voronoi_plot_2d_colored()`, which is in the `voronoi_plot_2d.py` utilities module, to plot higher-dimensional tessellations. Improve the visualization of a cell's mapped output class; currently, a cell's corresponding class is denoted by the color of the cell's centroid.