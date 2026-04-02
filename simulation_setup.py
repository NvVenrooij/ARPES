import numpy as np
import torch

class Simulation:

    def __init__(self, Material, simulation_input): 
        
        self.Material = Material
        self.simulation_input = simulation_input
        if self.simulation_input["is_plane"] != False:
            self.k_manifold = self.generate_k_plane()
            self.reciprocal_lattice = self.generate_reciprocal_lattice_plane()

        else:
            self.k_manifold = self.generate_k_path()
            self.reciprocal_lattice = self.generate_reciprocal_lattice_path()


    def generate_k_plane(self):

        """Generates an arbitrary plane through k-space set by the simulation input dictionary.
                    
            output: 
                - k_plane: NxNx3 tensor containing NxN k-vectors """

        kmin_x, kmax_x, kmin_z, kmax_z, k_y = self.simulation_input['k_ranges']
        point_1, point_2 = self.simulation_input['intersection_points'][:2]
        n_points = self.simulation_input['plane_density']
        lattice_constant = self.Material.material_parameters_AU['lattice_constant']


        if n_points%10 != 0:
            raise ValueError("n_points must be divisible by 10")
        
        n = torch.linspace(kmin_x, kmax_x, n_points+1)
        m = torch.linspace(kmin_z, kmax_z, n_points+1)
        M, N = torch.meshgrid(m,n)

        v1 = point_1/torch.norm(point_1)
        v2 = point_2 - torch.dot(point_2, v1) / torch.dot(v1, v1) * v1
        v2 /= torch.norm(v2)
        origin = torch.tensor([0,k_y,0])
        
        
        plane_points = origin + N[..., None] * v1 + M[..., None] * v2
        kx, ky, kz = plane_points[..., 0], plane_points[..., 1], plane_points[..., 2]
        k_plane = torch.stack([kx, ky, kz], dim=-1)*2*np.pi/lattice_constant
        
        return k_plane
    
    def generate_reciprocal_lattice_plane(self, n_layers=1, distance_threshold=None):
        """
        Generate reciprocal-lattice vectors that lie in the plane OR within a small
        distance from it ("neighboring zones").

        Args
        ----
        n_layers : int
            How many neighboring zone layers to include on each side of the plane
            (ignored if distance_threshold is provided).
        distance_threshold : float or None
            Direct thickness in bohr^-1. If None, computed from n_layers.

        Returns
        -------
        torch.Tensor of shape [M, 3]
            Unique reciprocal lattice vectors near the plane.
        """
        k_plane = self.k_manifold                            # [Ny, Nx, 3]
        r_vecs  = torch.stack(self.Material.reciprocal_lattice_vecs)  # [3, 3]
        device  = r_vecs.device
        dtype   = r_vecs.dtype

        # --- Robust plane normal from the grid (use corner + two edges) ---
        # Use top-left cell to avoid random/degenerate picks.
        p00 = k_plane[0, 0].to(dtype=dtype, device=device)
        # Handle degenerate 1xN or Nx1 just in case
        i1 = 1 if k_plane.shape[0] > 1 else 0
        j1 = 1 if k_plane.shape[1] > 1 else 0
        # If either dimension is 1, fallback to other valid differences
        v1 = (k_plane[i1, 0] - p00).to(dtype=dtype, device=device) if i1 != 0 else (k_plane[0, j1] - p00).to(dtype=dtype, device=device)
        v2 = (k_plane[0, j1] - p00).to(dtype=dtype, device=device) if j1 != 0 else (k_plane[i1, 0] - p00).to(dtype=dtype, device=device)

        # If the two edge vectors are (near) collinear, try a different pair
        if torch.norm(torch.cross(v1, v2)) < 1e-12:
            # fall back to two widely separated points
            pA = k_plane[k_plane.shape[0]//3, k_plane.shape[1]//3].to(dtype=dtype, device=device)
            pB = k_plane[2*k_plane.shape[0]//3, 2*k_plane.shape[1]//3].to(dtype=dtype, device=device)
            v1 = pA - p00
            v2 = pB - p00

        n_vec = torch.cross(v1, v2)
        n_norm = torch.norm(n_vec)
        if n_norm < 1e-14:
            # Degenerate plane; return empty
            return torch.empty((0, 3), dtype=dtype, device=device)
        n_hat = n_vec / n_norm

        # --- Determine coverage in G-space so we don’t miss neighbors ---
        # Get approximate magnitude range of the plane to decide how many G steps we need.
        # Use max |component| in k_plane as a simple bound.
        k_abs_max = torch.max(torch.abs(k_plane.to(dtype=dtype, device=device)))
        # Estimate how many steps of each reciprocal basis we need to cover that magnitude.
        # Add a small cushion (+2) so that neighbors are not clipped.
        r_norms = torch.stack([torch.norm(r_vecs[i]) for i in range(3)])
        steps_per_axis = torch.clamp((k_abs_max / r_norms).to(torch.long) + 2, min=2)
        max_steps = int(steps_per_axis.max().item())

        # --- Set distance threshold (plane thickness) ---
        if distance_threshold is None:
            # A simple heuristic: take a "typical" |G| and say layer spacing ~ |G| / 4
            # (FCC/diamond-like behavior; adjust if needed)
            typical_G = torch.mean(r_norms)
            layer_spacing = typical_G / 4.0
            distance_threshold = (n_layers + 0.5) * layer_spacing
        else:
            # Ensure it's a tensor on the right device/dtype
            distance_threshold = torch.as_tensor(distance_threshold, dtype=dtype, device=device)

        # --- Enumerate G and keep those near the plane ---
        keep = []
        eps = torch.as_tensor(1e-10, dtype=dtype, device=device)

        use_strict_plane = False

        if self.simulation_input["delta_layer"] != False:
            use_strict_plane = True  # or however you access it


        for i in range(-max_steps, max_steps + 1):
            for j in range(-max_steps, max_steps + 1):
                for k in range(-max_steps, max_steps + 1):
                    G = i * r_vecs[0] + j * r_vecs[1] + k * r_vecs[2]
                    dist = torch.abs(torch.dot(G, n_hat))

                    if use_strict_plane:
                        # delta-layer: only in-plane umklapp
                        if dist < eps:
                            keep.append(G)
                    else:
                        # bulk: slab for resolution / neighboring layers
                        if dist < distance_threshold:
                            keep.append(G)

        if not keep:
            return torch.empty((0, 3), dtype=dtype, device=device)

        Gs = torch.stack(keep, dim=0)
        # Deduplicate rows
        Gs_unique = torch.unique(Gs, dim=0)

        return Gs_unique
         
    def generate_standard_k_path(self):

        '''The standard reciprocal space path through all high symmetry points of the Brillouin Zone'''

        sim_params = self.Material.material_parameters_AU
        hsp = sim_params['high_symmetry_points']

        # Access high symmetry points
        L = hsp["L"].type(torch.float64)
        Gamma = hsp["Gamma"].type(torch.float64)
        Xz = hsp["Xz"].type(torch.float64)
        U = hsp["U"].type(torch.float64)
        K = hsp["K"].type(torch.float64)

        # Create paths
        Lambda = self.create_path(L, Gamma, 40)
        Delta = self.create_path(Gamma, Xz, 40)
        S = self.create_path(Xz, U, 10)
        Sigma = self.create_path(U, Gamma, 40)

        # Concatenate arrays for each dimension and ensure dtype is float64
        kx = torch.cat([Lambda[:, 0], Delta[:, 0], S[:, 0], Sigma[:, 0]]).type(torch.float64)
        ky = torch.cat([Lambda[:, 1], Delta[:, 1], S[:, 1], Sigma[:, 1]]).type(torch.float64)
        kz = torch.cat([Lambda[:, 2], Delta[:, 2], S[:, 2], Sigma[:, 2]]).type(torch.float64)

        LGXUKG = torch.stack([kx, ky, kz], dim=1).type(torch.float64)

        return LGXUKG
    
    def generate_k_path(self, start_zone=8.25, end_zone=17.5):
        """
        Generate a k-space path along the z-axis through multiple Brillouin zones.
        
        Parameters:
        -----------
        start_zone : float
            Starting point as multiple of Xz (default: -1)
        end_zone : float
            Ending point as multiple of Xz (default: 1)

        Examples:
        ---------
        # Path from 3*Xz to 10.5*Xz (your current case)
        path = self.generate_k_path(start_zone=3, end_zone=10.5)
        """

        
        sim_params = self.Material.material_parameters_AU
        hsp = sim_params['high_symmetry_points']
        n_points = self.simulation_input['plane_density']
        
        # Get Xz high-symmetry point
        Xz = hsp["Xz"].type(torch.float64)
        
        # Create the path
        start_point = start_zone * Xz
        end_point = end_zone * Xz  
        
        k_path = self.create_path(start_point, end_point, n_points)
        
        return k_path  # Already has the right shape and dtype
    
    def generate_reciprocal_lattice_path(self):
        '''Generates all the points on the reciprocal lattice that interscect the given line in k_space

            output:
                - tensor containing the reciprocal lattice vectors intersecting with the 
                line in k_space'''
        k_line = self.k_manifold
        r_vecs = torch.stack(self.Material.reciprocal_lattice_vecs)
        lattice_constant = self.Material.material_parameters_AU['lattice_constant']
        
        # For z-path: threshold for Gx and Gy
        threshold = 0.5 * (2*np.pi/lattice_constant)        
        k_max = int(torch.max(k_line))
        # Determine the number of steps in each direction
        combinations = []
        # Check wether dot product of reciprocal lattice vectors with random vector from k_line is zero
        for i in range(-k_max, k_max + 1):
            for j in range(-k_max, k_max + 1):
                for k in range(-k_max, k_max + 1):
                    G = (i * r_vecs[0] + j * r_vecs[1] + k * r_vecs[2]) 
                    if abs(G[0]) < threshold and abs(G[1]) < threshold:
                            combinations.append(G)        
        
        return torch.unique(torch.stack(combinations), dim=0) if combinations else torch.empty((0, 3))

    @staticmethod
    def create_path(start, end, n_points):
        """Linearly interpolate n_points between two 3D vectors."""
        t = torch.linspace(0, 1, n_points, dtype=torch.float64).unsqueeze(1)
        return start + t * (end - start)