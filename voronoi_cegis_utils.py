import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, _qhull
from dataclasses import dataclass
from maraboupy import MarabouCore

@dataclass
class Hyperplane:
    """A class that represents a hyperplane as a Cartesian equation."""

    coeffs: np.ndarray
    constant: float
    type: MarabouCore.Equation.EquationType

    def __str__(self):
        to_return = f"{self.coeffs[0]}*x0"
        for i in range(1, self.coeffs.size):
            to_return += f" + {self.coeffs[i]}*x{i}"

        to_return += " <= " if self.type == MarabouCore.Equation.LE else " >= "
        to_return += f"{self.constant}"
        return to_return

    def __repr__(self):
        return f'Hyperplane(\'{self.coeffs}\', {self.constant}\', {self.type})'

    def distance_to_hyperplane(self, point):
        unscaled_distance = np.dot(self.coeffs, point) + self.constant
        return unscaled_distance / np.linalg.norm(self.coeffs)

class VoronoiCell:
    """A class that represents a Voronoi tessellation with instances representing individual cells."""
    
    vor = None  # Voronoi tessellation object
    def __init__(self, centroid = (), ridges = [], output = 0, neighbors = []):
        self.centroid = centroid # Tuple of coordinate component floats
        self.ridges = ridges  # List of Hyerplane objects
        self.output = output  # Int
        self.neighbors = neighbors  # List of centroid indices

    def __str__(self):
        return f''

    def __repr__(self):
        return f'VoronoiCell(\'{self.centroid}\', {self.ridges}\', {self.output}\', {self.neighbors})'

    def compute_dist_to_farthest_ridge(self, point):
        return max([ridge.distance_to_hyperplane(point) for ridge in self.ridges])
    
    @staticmethod
    def compute_voronoi_tessellation(input_output_dict, add_points=False, centroid_cell_map={}):
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