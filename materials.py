import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

_DEFAULT_DB_PATH = Path(__file__).parent / 'materials_db.json'

class Material:
    
    """Semiconductor material for tight-binding and ARPES simulations.

    Loads parameters from materials_db.json, converts to atomic units
    (Hartree, Bohr), and builds the real-space lattice with nearest-
    neighbour information.
    """
    
    def __init__(self, name, **kwargs):
        self.name = name
        # Load parameters from JSON
        self.material_parameters_input = self.load_parameters(name)
        # Apply any overrides provided via kwargs
        self.material_parameters_input.update(kwargs)
        # Create a copy for simulation parameters
        self.simulation_parameters = {}
        # Convert parameters to atomic units and calculate Z_eff
        self.convert_to_atomic_units()
        # Generate the real space lattice
        self.real_space_lattice = self.generate_lattice()
        # Calculate list of neighbours
        self.neighbours = self.calc_neighbours()
        # Calculate the reciprocal lattice vectors
        self.reciprocal_lattice_vecs = self.calc_reciprocal_lattice_vecs()
    
    def convert_to_atomic_units(self):
        '''Converts the parameters of the material_db.json file from eV, nm
        to atomic units. Also calculates the effective charge.
        
        The JSON groups parameters by unit:
        - "lattice_constant_nm": converted nm → Bohr
        - "lattice_vectors", "basis": fractional → Bohr  
        - "high_symmetry_points": fractional → 1/Bohr
        - "energies_eV": all entries converted eV → Hartree
        '''

        bohr_radius_nm = 0.05291772105  # nm
        Hartree_in_eV = 27.2113862459   # eV

        params = self.material_parameters_input
        self.material_parameters_AU = {}

        # Lattice constant: nm → Bohr
        if 'lattice_constant_nm' not in params:
            raise ValueError(f"'lattice_constant_nm' not defined for material '{self.name}'.")
        lattice_constant = params['lattice_constant_nm'] / bohr_radius_nm
        self.material_parameters_AU['lattice_constant'] = lattice_constant

        # Lattice vectors and basis: fractional → Bohr
        for key in ['lattice_vectors', 'basis']:
            if key in params:
                arr = np.array(params[key]) * lattice_constant
                self.material_parameters_AU[key] = torch.tensor(arr, dtype=torch.float64)

        # High-symmetry points: fractional → 1/Bohr
        if 'high_symmetry_points' in params:
            hsp_converted = {}
            for key, value in params['high_symmetry_points'].items():
                k_frac = torch.tensor(value, dtype=torch.float64)
                hsp_converted[key] = k_frac * (2 * np.pi / lattice_constant)
            self.material_parameters_AU['high_symmetry_points'] = hsp_converted

        # Energies: eV → Hartree
        if 'energies_eV' in params:
            for key, value in params['energies_eV'].items():
                self.material_parameters_AU[key] = value / Hartree_in_eV

        # Derived quantities
        basis = self.material_parameters_AU.get('basis')
        if basis is None:
            raise ValueError("Basis positions are not defined.")

        d_nn = torch.norm(basis[1] - basis[0])
        self.material_parameters_AU['d_nn'] = d_nn


    @staticmethod
    def load_parameters(name, db_path=_DEFAULT_DB_PATH):
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Materials database not found: {db_path}")

        with open(db_path, 'r') as f:
            data = json.load(f)

        if name not in data:
            available = ', '.join(data.keys())
            raise ValueError(f"Material '{name}' not found. Available: {available}")
        return data[name]
                    
    def generate_lattice(self):
        """
        Generate positions of atoms in a lattice and store them in self.positions.

        Returns:
        - positions (torch.Tensor): Tensor of atomic positions.
        """
        lattice_vecs = self.material_parameters_AU["lattice_vectors"]
        basis = self.material_parameters_AU["basis"]
        n_cells = 3

        # Adjusted start and end to get n_cells along each axis
        start = - (n_cells - 1) // 2
        end = start + n_cells

        positions = []
        for n1 in torch.arange(start, end):
            for n2 in torch.arange(start, end):
                for n3 in torch.arange(start, end):
                    # Position of the origin of the current unit cell
                    R = n1 * lattice_vecs[0] + n2 * lattice_vecs[1] + n3 * lattice_vecs[2]
                    for b in basis:
                        # Calculate the absolute position of each basis atom
                        pos = R + b
                        positions.append(pos)
        positions = torch.stack(positions)
        # Remove duplicates due to floating-point precision
        positions_rounded = torch.round(positions * 1e8) / 1e8
        positions_unique = torch.unique(positions_rounded, dim=0)

        return positions_unique

    def plot_lattice(self):
        """
        Plot the lattice in 3D.
        """
        positions = self.real_space_lattice
        a = self.material_parameters_AU["lattice_constant"]
        n_cells = 3

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection='3d')

        # Plot the atoms
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c='blue', s=100, depthshade=True)

        # Set plot limits
        max_limit = a * n_cells
        ax.set_xlim(-max_limit / 2, max_limit / 2)
        ax.set_ylim(-max_limit / 2, max_limit / 2)
        ax.set_zlim(-max_limit / 2, max_limit / 2)

        # Labeling
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('Lattice Structure')

        plt.show()

    def calc_neighbours(self):
        """
        Calculate the positions of the first and second nearest neighbors
        of the central atom in the lattice.

        Returns:
        - first_nn_positions (torch.Tensor): Positions of first nearest neighbors.
        - second_nn_positions (torch.Tensor): Positions of second nearest neighbors.
        """
        positions = self.real_space_lattice

        # Step 1: Find the index of the central atom
        distances_to_origin = torch.norm(positions, dim=1)
        index_of_origin = torch.argmin(distances_to_origin).item()

        # Step 2: Calculate distances from the central atom to all other atoms
        distances = torch.norm(positions - positions[index_of_origin], dim=1)

        # Step 3: Sort the distances and get the sorted indices
        sorted_distances, sorted_indices = torch.sort(distances)

        # Remove zero distances (the central atom itself)
        nonzero_mask = sorted_distances > 1e-8
        sorted_distances = sorted_distances[nonzero_mask]
        sorted_indices = sorted_indices[nonzero_mask]

        # Ensure there are enough unique distances
        unique_distances = torch.unique(sorted_distances)
        if unique_distances.numel() < 2:
            raise ValueError("Not enough unique distances to determine nearest neighbors.")

        # Step 5: Find indices corresponding to the first and second nearest neighbors
        first_nn_distance = unique_distances[0]
        first_nn_mask = torch.isclose(sorted_distances, first_nn_distance)
        first_nn_indices = sorted_indices[first_nn_mask]

        second_nn_distance = unique_distances[1]
        second_nn_mask = torch.isclose(sorted_distances, second_nn_distance)
        second_nn_indices = sorted_indices[second_nn_mask]

        # Step 6: Get the positions of the nearest neighbors
        first_nn_positions = positions[first_nn_indices]
        second_nn_positions = positions[second_nn_indices]

        return first_nn_positions, second_nn_positions


    def calc_reciprocal_lattice_vecs(self):
        """
        Calculates the reciprocal lattice vectors given the real lattice vectors.

        Returns:
        - reciprocal_vectors (tuple): Tuple containing 3 torch.Tensors of shape (3,)
        """
        basis_vecs = self.material_parameters_AU["lattice_vectors"]

        # Volume of the unit cell
        V = torch.dot(basis_vecs[0], torch.cross(basis_vecs[1], basis_vecs[2]))

        b1 = 2 * np.pi * torch.cross(basis_vecs[1], basis_vecs[2]) / V
        b2 = 2 * np.pi * torch.cross(basis_vecs[2], basis_vecs[0]) / V
        b3 = 2 * np.pi * torch.cross(basis_vecs[0], basis_vecs[1]) / V

        return (b1, b2, b3)
