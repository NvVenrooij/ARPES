import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from matplotlib.colors import LinearSegmentedColormap, LogNorm


class ARPES:
    def __init__(self, Material, Tight_binding, Simulation_setup):
        self.Material = Material
        self.Tight_binding = Tight_binding
        self.Simulation_setup = Simulation_setup
        
        # Only compute plane intensity for 2D planes
        if self.Simulation_setup.simulation_input.get('is_plane', False):
            if self.Simulation_setup.simulation_input.get('delta_layer', True):
                self.intensity_plane, self.MME = self.Intensity_plane_delta_layer_from_scratch()
            else:
                self.intensity_plane = self.Intensity_plane_bulk()
        else:
            self.intensity_path = self.Intensity_path_bands()

    def Phi310(self, k_vecs):

        """Calculates the values of the Fourier transformed 3pz-orbital.
        
            input:
                - k_vecs: a 3xN or 3xNxN tensor containing all k-space vectors on the 
                path
                
            output: 
                - a 3xN or 3xNxN tensor containing all values for the
                Fourier transform at every k-vector in k_vecs on the path"""


        Z = self.Material.material_parameters_AU['effective_nuclear_charge']

        C = 1/(486*np.sqrt(6))
        k_mag = torch.sqrt(torch.sum(k_vecs**2,dim=-1))
        return C*4*np.pi*1J*0.5*np.sqrt(3/np.pi)*(k_vecs[..., 2]) * (
            4374*(4*Z**2-9*k_mag**2)/(Z**7*((1+9*k_mag**2/Z**2)**(7/2))) - 10935*(4*Z**2-27*k_mag**2)/(Z**8*((1+9*k_mag**2/Z**2)**(9/2)))
        )

    def Phi311(self, k_vecs):
        
        """Calculates the values of the Fourier transformed 3px-orbital
        
            input:
                - k_vecs: a 3xN or 3xNxN tensor containing all k-space vectors on the 
                path

            output: 
                - a 3xN or 3xNxN tensor containing all values for the
                Fourier transform at every k-vector in k_vecs on the path"""

        Z = self.Material.material_parameters_AU['effective_nuclear_charge']

        C = 1/(486*np.sqrt(6))
        k_mag = torch.sqrt(torch.sum(k_vecs**2,dim=-1))
        return C*4*np.pi*1J*0.5*np.sqrt(3/(2*np.pi))*(k_vecs[..., 0] + k_vecs[..., 1]*1J) * (
            4374*(4*Z**2-9*k_mag**2)/(Z**7*((1+9*k_mag**2/Z**2)**(7/2))) - 10935*(4*Z**2-27*k_mag**2)/(Z**8*((1+9*k_mag**2/Z**2)**(9/2)))
        )

    def Phi31m1(self, k_vecs):
        
        """Calculates the values of the Fourier transformed 3py-orbital.
        
            input:
                - k_vecs: a 3xN or 3xNxN tensor containing all k-space vectors on the 
                path

            output: 
                - a 3xN or 3xNxN tensor containing all values for the
                Fourier transform at every k-vector in k_vecs on the path"""

        Z = self.Material.material_parameters_AU['effective_nuclear_charge']

        C = 1/(486*np.sqrt(6))
        k_mag = torch.sqrt(torch.sum(k_vecs**2,dim=-1))
        return C*4*np.pi*1J*0.5*np.sqrt(3/(2*np.pi))*(k_vecs[..., 0] - k_vecs[..., 1]*1J) * (
            4374*(4*Z**2-9*k_mag**2)/(Z**7*((1+9*k_mag**2/Z**2)**(7/2))) - 10935*(4*Z**2-27*k_mag**2)/(Z**8*((1+9*k_mag**2/Z**2)**(9/2)))
        )

    def Phi300(self, k_vecs):

        """Calculates the Fourier transform of the 3s-orbital.
        
            input:
                - k_vecs: a 3xN or 3xNxN tensor containing all k-space vectors on the 
                path

            output: 
                - a 3xN or 3xNxN tensor containing all values for the
                Fourier transform at every k-vector in k_vecs on the path"""
        
        Z = self.Material.material_parameters_AU['effective_nuclear_charge']

        k_mag = torch.sqrt(torch.sum(k_vecs**2,dim=-1))
        D = np.sqrt(1/(3*np.pi)) * 1/(486)
        I = 729*(2*Z**2 - 9*k_mag**2)/(Z**5*(1+9*k_mag**2/Z**2)**(5/2))
        II = -4274*(2*Z**2 - 27*k_mag**2)/(Z**5*(1+9*k_mag**2/Z**2)**(7/2))
        III = 1458*(8*Z**4 - 216*k_mag**2*Z**2 + 243*k_mag**4)/(Z**7*(1+9*k_mag**2/Z**2)**(9/2))
        return D*4*np.pi*1/(2*np.sqrt(np.pi)) * (I + II + III)

    def Phi400(self, k_vecs):

        """Calculates the Fourier transform of the 4s(s*) -orbital.
        
            input:
                - k_vecs: a 3xN or 3xNxN tensor containing all k-space vectors on the 
                path

            output: 
                - a 3xN or 3xNxN tensor containing all values for the
                Fourier transform at every k-vector in k_vecs on the path"""
        
        Z = self.Material.material_parameters_AU['effective_nuclear_charge']

        k_mag = torch.sqrt(torch.sum(k_vecs**2,dim=-1))
        D = 1/(486*np.sqrt(6)) * 27/256 * np.sqrt(3/2)
        I = 12288 * Z**(-3) * (1 - 8 * k_mag**2 / Z**2) / (1 + 16 * k_mag**2 / Z**2)**(5/2)
        II = -110592 * Z**(-3) * (1 - 24 * k_mag**2 / Z**2) / (1 + 16 * k_mag**2 / Z**2)**(7/2)
        III = 294912 * Z**(-3) * (1 - 48 * k_mag**2 / Z**2 + 96 * k_mag**4 / Z**4) / (1 + 16 * k_mag**2 / Z**2)**(9/2)
        IV = -245760 * Z**(-3) * (1 - 80 * k_mag**2 / Z**2 + 480 * k_mag**4 / Z**4) / (1 + 16 * k_mag**2 / Z**2)**(11/2)
        return D*4*np.pi*1/(2*np.sqrt(np.pi)) * (I + II + III + IV)


    def k_dot_lambda(self, k_vecs):

        '''returns the dot product between the momentum plane and the polarization vector
        
            input:
                - k_vecs: a 3xN or 3xNxN tensor containing all k-space vectors on the 
                path

            output:
                - linear combination between the momenta in k_vecs determined by the chosen polarization
        '''

        pol = self.Simulation_setup.simulation_input['polarization']

        if pol == 'x':
            return k_vecs[..., 0]
        if pol == 'y':
            return k_vecs[..., 1]
        if pol == 'z':
            return k_vecs[..., 2]
        if pol == 'kx_ky': #rcp in xy plane, light propagates along z
            return (k_vecs[..., 0]+1J*k_vecs[..., 1])/np.sqrt(2)
        if pol == 'kx_kz': #rcp in xz plane, light propagates along y
            return (k_vecs[..., 2]+1J*k_vecs[..., 0])/np.sqrt(2)
        if pol == 'ky_kz': #rcp in yz plane, light propagates along x
            return (k_vecs[..., 1]+1J*k_vecs[..., 2])/np.sqrt(2)
        if pol == 'ky_kx': #rcp in xy plane, light propagates along z
            return (k_vecs[..., 0]-1J*k_vecs[..., 1])/np.sqrt(2)
        if pol == 'kz_kx': #rcp in xz plane, light propagates along y
            return (k_vecs[..., 2]-1J*k_vecs[..., 0])/np.sqrt(2)
        if pol == 'kz_ky': #rcp in yz plane, light propagates along x
            return (k_vecs[..., 1]-1J*k_vecs[..., 2])/np.sqrt(2)
        
    def orbital_matrix_element(self, evec, phiS, phiZ, phiX1, phiX2, phiSs, WA, WB, kdpm):
        """
        Contract TB eigenvectors with orbital Fourier factors and sublattice weights.

        Parameters
        ----------
        evec : torch.Tensor, shape (N, dim)
            Eigenvector components for one band on the selected k-points.
        phiS, phiZ, phiX1, phiX2, phiSs : torch.Tensor, shape (N,)
            Orbital Fourier factors evaluated at the full photoelectron momentum q.
        WA, WB : torch.Tensor, shape (N,)
            Sublattice weights / phases for A and B sites.
            For bulk, pass arrays of ones.
        kdpm : torch.Tensor, shape (N,)
            Polarization prefactor eps·q on the selected points.

        Returns
        -------
        torch.Tensor, shape (N,)
            Complex matrix element contribution for this band.
        """
        use_soc   = bool(self.Simulation_setup.simulation_input.get("spin_orbit", False))

        idx = self.Tight_binding.basis_index_sp3s

        if use_soc:
            idx_soc = self.Tight_binding.basis_index_sp3s_soc

        expected_dim = 20 if use_soc else 10 
        if evec.shape[1] < expected_dim:
            raise ValueError(
                f"Eigenvector dimension {evec.shape[1]} is smaller than expected {expected_dim} "
                f"for spin_orbit={use_soc}."
            )
        
        Xfac = (phiX1 + phiX2) / np.sqrt(2)
        Yfac = -1j * (phiX1 - phiX2) / np.sqrt(2)

        S = phiS * (WA * evec[:, idx["A_s"]]  + WB * evec[:, idx["B_s"]])
        X = Xfac * (WA * evec[:, idx["A_px"]] + WB * evec[:, idx["B_px"]])
        Y = Yfac * (WA * evec[:, idx["A_py"]] + WB * evec[:, idx["B_py"]])
        Z = phiZ * (WA * evec[:, idx["A_pz"]] + WB * evec[:, idx["B_pz"]])
        Ss = phiSs * (WA * evec[:, idx["A_sstar"]] + WB * evec[:, idx["B_sstar"]])
        M = kdpm * (S + X + Y + Z + Ss)

        if use_soc:
            S_v = phiS * (WA * evec[:, idx_soc["A_s_dn"]]  + WB * evec[:, idx_soc["B_s_dn"]])
            X_v = Xfac * (WA * evec[:, idx_soc["A_px_dn"]] + WB * evec[:, idx_soc["B_px_dn"]])
            Y_v = Yfac * (WA * evec[:, idx_soc["A_py_dn"]] + WB * evec[:, idx_soc["B_py_dn"]])
            Z_v = phiZ * (WA * evec[:, idx_soc["A_pz_dn"]] + WB * evec[:, idx_soc["B_pz_dn"]])
            Ss_v = phiSs * (WA * evec[:, idx_soc["A_sstar_dn"]] + WB * evec[:, idx_soc["B_sstar_dn"]])

            M = M + kdpm * (S_v + X_v + Y_v + Z_v + Ss_v)

        return M

    def Intensity_plane_delta_layer(self):
        """
        Works fine for the XxXz plane.
        For the KXz plane, use Intensity_plane_delta_layer_from_scratch_new.
        """
        k_plane = self.Simulation_setup.k_manifold
        bands   = self.Tight_binding.bands
        evecs   = self.Tight_binding.evecs
        G_all   = self.Simulation_setup.reciprocal_lattice

        Ni, Nj, n_bands = bands.shape
        device = k_plane.device
        MME = torch.zeros((Ni, Nj), dtype=torch.complex128, device=device)

        E0      = self.Simulation_setup.simulation_input["energy"] / 27.2113862459
        sigma_E = self.Simulation_setup.simulation_input["bandwidth"] / 27.2113862459

        a = self.Material.material_parameters_AU["lattice_constant"]
        measure = 2 * np.pi / a
        kz_shift = self.Simulation_setup.simulation_input["kz_shift"] * measure

        eps = 1e-10
        G_perp = G_all[(torch.abs(G_all[:, 0]) < eps) & (torch.abs(G_all[:, 1]) < eps)]

        if G_perp.numel() == 0:
            raise ValueError("Need non-empty G_perp (Gx≈Gy≈0).")

        phi_funcs = [self.Phi310, self.Phi311, self.Phi31m1, self.Phi300, self.Phi400]

        basis = self.Material.material_parameters_AU["basis"].to(device=device, dtype=k_plane.dtype)
        delta_tau = basis[1] - basis[0]

        kz_meas = k_plane[..., 2]
        W0 = torch.zeros((Ni, Nj), dtype=torch.complex128, device=device)
        W1 = torch.zeros((Ni, Nj), dtype=torch.complex128, device=device)

        for gp in G_perp:
            gpz = gp[2]

            k_tmp = k_plane.clone()
            k_tmp[..., 2] = kz_meas - gpz - kz_shift
            F_tot = self.envelope_finite_square_well(k_tmp)

            phase_B = torch.exp(-1j * gpz * delta_tau[2])

            W0 += F_tot
            W1 += F_tot * phase_B

        phi_shifted = [phi_func(k_plane) for phi_func in phi_funcs]
        kdp = self.k_dot_lambda(k_plane)

        for b in range(n_bands):
            bandE = bands[:, :, b]
            w = torch.exp(-(bandE - E0)**2 / (2 * sigma_E**2))
            mask = w > 1e-4
            if not torch.any(mask):
                continue

            Mi, Mj = torch.where(mask)
            evec = evecs[Mi, Mj, :, b]

            M = self.orbital_matrix_element(
                evec=evec,
                phiS=phi_shifted[3][Mi, Mj],
                phiZ=phi_shifted[0][Mi, Mj],
                phiX1=phi_shifted[1][Mi, Mj],
                phiX2=phi_shifted[2][Mi, Mj],
                phiSs=phi_shifted[4][Mi, Mj],
                WA=W0[Mi, Mj],
                WB=W1[Mi, Mj],
                kdpm=kdp[Mi, Mj],
            )

            MME[Mi, Mj] += M * w[Mi, Mj]

        intensity = torch.abs(MME)**2
        mx = intensity.max()
        if mx > 0:
            intensity = intensity / mx
        return intensity, MME    
    
    def Intensity_plane_bulk(self):
            '''Calculates the intensity for electrons excited within a pre-defined energy bandwidth
            using Gaussian energy broadening for smooth intensity profiles.
            
            Physics:
                The matrix element is M(q) = sum_G sum_{a,v} a_{av}(k) * phi_tilde(q) * (eps . q)
                where q is the full photoelectron momentum and k = q - G is the crystal momentum.
                
                phi_tilde(q) is the FT of the atomic orbital evaluated at the FULL q, not shifted 
                by G. It is centered at q=0 and provides the smooth cross-BZ intensity envelope
                reflecting the diminishing high-momentum Fourier content of the bound state.
                
                The eigenvector coefficients a(k) and structure factors are periodic in G and 
                produce the repeating in-zone pattern.

            output: tensor of shape NxN containing the intensity at each index in the k_plane tensor
            '''

            # Unpack parameters
            k_vecs = self.Simulation_setup.k_manifold            # These are q-vectors (full photoelectron momenta)
            bands = self.Tight_binding.bands
            evecs = self.Tight_binding.evecs
            reciprocal_vecs = self.Simulation_setup.reciprocal_lattice

            Ni, Nj, n_bands = bands.shape
            MME = torch.zeros((Ni, Nj), dtype=torch.complex128)
            n_vecs = reciprocal_vecs.shape[0]
            E = self.Simulation_setup.simulation_input["energy"] / 27.2113862459
            sigma_E = self.Simulation_setup.simulation_input["bandwidth"] / 27.2113862459  # Gaussian width
            phi_funcs = [self.Phi310, self.Phi311, self.Phi31m1, self.Phi300, self.Phi400]

            # phi_tilde(q): Fourier transform of the isolated atomic orbital, evaluated at the
            # FULL photoelectron momentum q. This is computed once outside the G-loop because
            # it does not depend on G — it is a property of the single atom, centered at q=0.
            # Its slow decay at large |q| provides the cross-BZ intensity envelope.
            phi_at_q = [phi_func(k_vecs) for phi_func in phi_funcs]

            # eps . q: the polarization prefactor also uses the full photoelectron momentum
            kdp_at_q = self.k_dot_lambda(k_vecs)

            print(f"Calculating intensity plane with Gaussian broadening (σ = {sigma_E*27.211:.3f} eV)...")

            for b in range(n_bands):
                # Get band energies at all points in the plane
                band_energies = bands[:, :, b]  # Shape: (Ni, Nj)
                
                # Calculate Gaussian weights based on energy difference
                energy_diff = band_energies - E
                weights = torch.exp(-energy_diff**2 / (2 * sigma_E**2))  # Shape: (Ni, Nj)
                
                # Only process points with significant weight (3-sigma cutoff)
                significant_mask = weights > 1e-4
                
                if not torch.any(significant_mask):
                    continue
                
                Mi, Mj = torch.where(significant_mask)
                evec = evecs[Mi, Mj, :, b]     

                ones = torch.ones_like(kdp_at_q[Mi, Mj], dtype=torch.complex128)

                MME_contribution = self.orbital_matrix_element(
                    evec=evec,
                    phiS=phi_at_q[3][Mi, Mj],
                    phiZ=phi_at_q[0][Mi, Mj],
                    phiX1=phi_at_q[1][Mi, Mj],
                    phiX2=phi_at_q[2][Mi, Mj],
                    phiSs=phi_at_q[4][Mi, Mj],
                    WA=ones,
                    WB=ones,
                    kdpm=kdp_at_q[Mi, Mj],
                )

                # Apply Gaussian weighting and accumulate
                MME[Mi, Mj] += MME_contribution * weights[Mi, Mj]

            # Calculate photoemission intensity
            intensity = torch.abs(MME)**2
            
            # Normalize to maximum intensity
            max_intensity = torch.max(intensity)
            if max_intensity > 0:
                intensity /= max_intensity

            return intensity
        

    def Intensity_plane_delta_layer_from_scratch(
        self,
        keep_gy_strips=1,
        gy_tol=1e-8,
        w_mask_min=1e-2,
        energy_mask_min=1e-4
    ):
        q_plane = self.Simulation_setup.k_manifold
        Ni, Nj, _ = q_plane.shape
        device = q_plane.device
        dtypeR = q_plane.dtype

        E0      = self.Simulation_setup.simulation_input["energy"]    / 27.2113862459
        sigma_E = self.Simulation_setup.simulation_input["bandwidth"] / 27.2113862459

        a        = self.Material.material_parameters_AU["lattice_constant"]
        measure  = 2 * np.pi / a
        k_target = self.Simulation_setup.simulation_input["kz_shift"] * measure

        use_soc = bool(self.Simulation_setup.simulation_input.get("spin_orbit", False))

        G_all = self.Simulation_setup.reciprocal_lattice.to(device=device, dtype=dtypeR)

        basis = self.Material.material_parameters_AU["basis"].to(device=device, dtype=dtypeR)
        tauA, tauB = basis[0], basis[1]

        phi_funcs = [self.Phi310, self.Phi311, self.Phi31m1, self.Phi300, self.Phi400]
        phi_q     = [pf(q_plane) for pf in phi_funcs]
        kdp_q     = self.k_dot_lambda(q_plane)
        qz        = q_plane[..., 2]

        q_flat   = q_plane.reshape(-1, 3)
        kdp_flat = kdp_q.reshape(-1)
        phi_flat = [p.reshape(-1) for p in phi_q]

        MME = torch.zeros((Ni, Nj), dtype=torch.complex128, device=device)
        MME_flat = MME.reshape(-1)

        if keep_gy_strips is None:
            G_list = G_all
        else:
            absGy = torch.abs(G_all[:, 1])
            nonzero = absGy > gy_tol

            if torch.any(nonzero):
                Gy_min = torch.min(absGy[nonzero])
                m = torch.round(G_all[:, 1] / Gy_min)
                Gy_on_strip = torch.abs(G_all[:, 1] - m * Gy_min) <= gy_tol
                m_ok = torch.abs(m) <= keep_gy_strips
                G_list = G_all[Gy_on_strip & m_ok]
            else:
                G_list = G_all[torch.abs(G_all[:, 1]) <= gy_tol]

        for G in G_list:
            Gx, Gy, Gz = G

            phaseA = torch.exp(-1j * torch.dot(G, tauA))
            phaseB = torch.exp(-1j * torch.dot(G, tauB))

            q_env = q_plane.clone()
            q_env[..., 2] = qz - k_target - Gz
            F = self.envelope_finite_square_well(q_env)

            WA_full = F * phaseA
            WB_full = F * phaseB

            mask0 = (torch.abs(WA_full) + torch.abs(WB_full)).real > w_mask_min
            if not torch.any(mask0):
                continue

            idx   = torch.where(mask0.reshape(-1))[0]
            q_sel = q_flat[idx]

            k_sel = q_sel - G[None, :]
            H_sel = self.Tight_binding.sp3s(spin_orbit=use_soc, k_manifold=k_sel)
            evals, evecs = torch.linalg.eigh(H_sel)

            WA = WA_full.reshape(-1)[idx]
            WB = WB_full.reshape(-1)[idx]
            kdpm = kdp_flat[idx]

            phiS  = phi_flat[3][idx]
            phiZ  = phi_flat[0][idx]
            phiX1 = phi_flat[1][idx]
            phiX2 = phi_flat[2][idx]
            phiSs = phi_flat[4][idx]

            for b in range(evals.shape[1]):
                bandE = evals[:, b].real
                w = torch.exp(-(bandE - E0)**2 / (2 * sigma_E**2))

                hit = w > energy_mask_min
                if not torch.any(hit):
                    continue

                ids  = torch.where(hit)[0]
                evec = evecs[ids, :, b]

                MME_contrib = self.orbital_matrix_element(
                    evec=evec,
                    phiS=phiS[ids],
                    phiZ=phiZ[ids],
                    phiX1=phiX1[ids],
                    phiX2=phiX2[ids],
                    phiSs=phiSs[ids],
                    WA=WA[ids],
                    WB=WB[ids],
                    kdpm=kdpm[ids],
                )

                MME_flat[idx[ids]] += MME_contrib * w[ids]

        intensity = torch.abs(MME) ** 2
        mx = intensity.max()
        if mx > 0:
            intensity /= mx

        return intensity, MME
    
    def envelope_finite_square_well(self, k_vecs: torch.Tensor) -> torch.Tensor:
        # --- 1) grab parameters & slice ---
        V0 = self.Simulation_setup.simulation_input['potential well depth']/ 27.2113862459
        L  = self.Simulation_setup.simulation_input['potential well width'] * self.Material.material_parameters_AU['lattice_constant']
        kz = k_vecs[..., 2]

        # --- 2) solve transcendental & lift to torch ---
        k_tilde, kappa_tilde = self.solve_transcendental(V0, L, parity='even', solution_index=0)
        dtype, device = kz.dtype, kz.device
        k_t   = torch.tensor(k_tilde,   dtype=dtype, device=device)
        κ_t   = torch.tensor(kappa_tilde, dtype=dtype, device=device)
        L_t   = torch.tensor(L,         dtype=dtype, device=device)

        # --- 3) normalization constants ---
        A = 1/torch.sqrt(
            torch.cos(k_t * L_t/2)**2 / κ_t
            + L_t/2
            + torch.sin(k_t * L_t) / (2*k_t)
        )
        B = A * torch.cos(k_t * L_t/2) * torch.exp(κ_t * L_t/2)

        num_1 = 2 * B * torch.exp(-L * κ_t/2) * (κ_t * torch.cos(kz * L/2) - kz * torch.sin(kz * L/2))

        den_1 = (kz**2 + κ_t**2)

        num_2 = 2 * A * (k_t * torch.cos(kz * L/2) * torch.sin(k_t * L/2) - kz * torch.sin(kz * L/2) * torch.cos(k_t * L/2))

        den_2 = (k_t**2 - kz**2)

        # --- 5) return sum of shifted branches ---
        return num_1/den_1 + num_2/den_2
    
        
    def solve_transcendental(self, V0, L, parity='even', solution_index=0):
        """
        Solve the transcendental equation for finite square well.

        Parameters:
        - V0: Potential depth (Hartree)
        - L: Width of the potential well (bohr radius)
        - parity: 'even' or 'odd'
        - solution_index: integer, specifies which solution to pick (0-based indexing)

        Returns:
        - eta: solution of the transcendental equation
        - k: wave vector inside the well
        - kappa: decay constant outside the well
        """

        # Compute zeta
        zeta = (L / 2) * np.sqrt(2 * V0)

        # Define transcendental equations
        def transcendental_even(eta):
            if eta <= 0 or eta >= zeta:
                return np.inf
            return np.tan(eta) - np.sqrt((zeta / eta)**2 - 1)

        def transcendental_odd(eta):
            if eta <= 0 or eta >= zeta:
                return np.inf
            return -1/np.tan(eta) - np.sqrt((zeta / eta)**2 - 1)

        # Select appropriate equation based on parity
        equation = transcendental_even if parity == 'even' else transcendental_odd

        # Determine approximate intervals based on multiples of pi/2
        intervals = []
        n_intervals = int(zeta / (np.pi / 2))
        for n in range(n_intervals):
            start = n * np.pi / 2 + 1e-5
            end = (n + 1) * np.pi / 2 - 1e-5
            if start < zeta:
                intervals.append((start, min(end, zeta - 1e-5)))

        # Find all solutions within intervals
        solutions = []
        for interval in intervals:
            try:
                sol = root_scalar(equation, bracket=interval, method='brentq')
                if sol.converged:
                    solutions.append(sol.root)
            except ValueError:
                continue  # Skip intervals without roots

        if not solutions:
            raise ValueError("No solutions found with given parameters and parity.")

        if solution_index >= len(solutions):
            raise IndexError(f"Requested solution_index {solution_index} exceeds number of found solutions {len(solutions)}.")

        # Select specified solution
        eta_selected = solutions[solution_index]
        k = 2 * eta_selected / L
        kappa = k * np.sqrt(zeta**2/eta_selected**2 - 1) 

        return k, kappa


    def plot_intensity_plane(self, intensity=None):

        k_vecs = self.Simulation_setup.k_manifold
        lattice_constant = self.Material.material_parameters_AU["lattice_constant"]
        if intensity == None:
            intensity_array = self.intensity_plane
        E = self.Simulation_setup.simulation_input["energy"]
        v_max = self.Simulation_setup.simulation_input["v_max"]
        #orientation = self.Simulation_setup.simulation_input["plane_orientation"]
        pol = self.Simulation_setup.simulation_input['polarization']
        plane = self.Simulation_setup.simulation_input["intersection_points"][-1]
        grid = self.Simulation_setup.simulation_input["grid"]

        # Extract the max and min k values in each direction and convert to units of 2pi/a
        kmax_x, kmax_y, kmax_z = (k_vecs[:,:,0].max()*lattice_constant/(2*np.pi), 
                                k_vecs[:,:,1].max()*lattice_constant/(2*np.pi), 
                                k_vecs[:,:,2].max()*lattice_constant/(2*np.pi))
        
        kmin_x, kmin_y, kmin_z = (k_vecs[:,:,0].min()*lattice_constant/(2*np.pi), 
                                k_vecs[:,:,1].min()*lattice_constant/(2*np.pi), 
                                k_vecs[:,:,2].min()*lattice_constant/(2*np.pi))
        
        #kmin_x, kmax_x, kmin_z, kmax_z, k_y = self.Simulation_setup.simulation_input['k_ranges']

        # Determine the axis limits and labels based on the chosen orientation
        # IMPORTANT: In this ARPES experiment the data will always be shown 
        #            in the kx-kz plane.
        
        # Define the W points if the grid condition matches
        if plane == 'XxXz' and grid == True:
            # Generate reciprocal vectors if needed
            reciprocal_vecs = self.Simulation_setup.reciprocal_lattice * lattice_constant / (2 * np.pi)
            n_vecs = reciprocal_vecs.shape[0]

            # Define W points using torch tensors
            W1 = torch.tensor([1, 0, 0.5])
            W2 = torch.tensor([0.5, 0, 1])
            W3 = torch.tensor([-0.5, 0, 1])
            W4 = torch.tensor([-1, 0, 0.5])
            W5 = torch.tensor([-1, 0, -0.5])
            W6 = torch.tensor([-0.5, 0, -1])
            W7 = torch.tensor([0.5, 0, -1])
            W8 = torch.tensor([1, 0, -0.5])
            Ws = torch.stack([W1, W2, W3, W4, W5, W6, W7, W8])

            scaling = 1

            # Loop through each reciprocal vector and shift Ws temporarily for plotting
            for v in range(n_vecs):
                shift = reciprocal_vecs[v]
                shifted_Ws = Ws + shift  # Apply shift temporarily
                
                # Convert to numpy for compatibility with matplotlib
                shifted_Ws = shifted_Ws.numpy()

                # Draw lines between consecutive W points
                for i in range(len(shifted_Ws)):
                    x1, y1 = shifted_Ws[i - 1, 0], shifted_Ws[i - 1, 2]
                    x2, y2 = shifted_Ws[i, 0], shifted_Ws[i, 2]
                    
                    # Plot line connecting the points
                    plt.plot([x1, x2], [y1, y2], color='white', linewidth=1)

        if plane == 'KXz' and grid == True:
            # Generate reciprocal vectors if needed
            reciprocal_vecs = self.Simulation_setup.reciprocal_lattice * lattice_constant / (2 * np.pi)
            n_vecs = reciprocal_vecs.shape[0]

            Gamma_X = 1
            U_X = np.sqrt(1/8)
            Gamma_K = 1.5/np.sqrt(2)

            # Define W points using torch tensors
            W1 = torch.tensor([U_X, 0, Gamma_X])
            W2 = torch.tensor([-U_X, 0, Gamma_X])
            W3 = torch.tensor([-Gamma_K, 0, 0])
            W4 = torch.tensor([-U_X, 0, -Gamma_X])
            W5 = torch.tensor([U_X, 0, -Gamma_X])
            W6 = torch.tensor([Gamma_K, 0, 0])
            Ws = torch.stack([W1, W2, W3, W4, W5, W6])

            scaling = np.sqrt(2)

            # Loop through each reciprocal vector and shift Ws temporarily for plotting
            for v in range(n_vecs):
                shift = reciprocal_vecs[v]*torch.tensor([scaling,scaling,1])
                shifted_Ws = Ws + shift  # Apply shift temporarily
                
                # Convert to numpy for compatibility with matplotlib
                shifted_Ws = shifted_Ws.numpy()

                # Draw lines between consecutive W points
                for i in range(len(shifted_Ws)):
                    x1, y1 = shifted_Ws[i - 1, 0], shifted_Ws[i - 1, 2]
                    x2, y2 = shifted_Ws[i, 0], shifted_Ws[i, 2]
                    
                    # Plot line connecting the points
                    plt.plot([x1, x2], [y1, y2], color='white', linewidth=1)

    # Define your custom colormap
        colors = ["black", "red", "yellow"]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

        # Display the intensity array using imshow
        plt.imshow(
            intensity_array,
            extent=[scaling*kmin_x, scaling*kmax_x, kmin_z, kmax_z],
            origin='lower',
            vmax=v_max,
            cmap=cmap,
            interpolation='gaussian'
        )

        # Get current axes
        ax = plt.gca()

        # Set axis labels with larger font size and bold font weight
        ax.set_xlabel('kx ($2\pi$/a)', fontsize=14)
        ax.set_ylabel('kz ($2\pi$/a)', fontsize=14)

        # Increase the font size of tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)

    
    def Intensity_path_bands(self):
        """Calculates the ARPES intensity along a k-path as function of energy.
        
        Uses Gaussian energy broadening for smooth intensity profiles.
        
        Physics:
            phi_tilde(q) is evaluated at the full photoelectron momentum q (not shifted by G).
            It is a property of the isolated atom, centered at q=0, and provides the smooth
            cross-BZ intensity envelope. The eigenvectors a(k) use the reduced crystal 
            momentum k = q - G, and are periodic in G by construction.
        
        Returns:
            intensity_2d: torch.Tensor of shape (n_energies, n_kpoints)
            Erange: numpy array of energy values in Hartree  
            k_vecs: k-point coordinates
        """
        
        # Unpack parameters
        k_vecs = self.Simulation_setup.k_manifold  # These are q-vectors (full photoelectron momenta)
        bands = self.Tight_binding.bands  # Shape: (N_k, n_bands)
        evecs = self.Tight_binding.evecs  # Shape: (N_k, dim, n_bands)
        reciprocal_vecs = self.Simulation_setup.reciprocal_lattice
        
        N_k, n_bands = bands.shape
        n_vecs = reciprocal_vecs.shape[0]
        
        # Energy range setup
        Emax = 0.1/27.2113862459
        Emin = -6.2/27.2113862459
        step = (Emax - Emin) / N_k
        Erange = np.arange(Emin, Emax, step)
        
        # Gaussian broadening width (convert from eV to Hartree)
        sigma_E = self.Simulation_setup.simulation_input["bandwidth"] / 27.2113862459
        
        # Initialize intensity array
        intensity_2d = torch.zeros((len(Erange), N_k), dtype=torch.float64)
        
        # Phi functions for Fourier transforms of atomic orbitals
        phi_funcs = [self.Phi310, self.Phi311, self.Phi31m1, self.Phi300, self.Phi400]
        
        # phi_tilde(q): evaluated at the FULL photoelectron momentum q, computed once
        # outside the G-loop. This does not depend on G — it reflects the momentum-space
        # content of the isolated atomic orbital and produces the cross-BZ envelope.
        phi_at_q = [phi_func(k_vecs) for phi_func in phi_funcs]
        
        # eps . q: polarization prefactor at full photoelectron momentum
        kdp_at_q = self.k_dot_lambda(k_vecs)
        
        print(f"Calculating intensity map with Gaussian broadening (σ = {sigma_E*27.211:.3f} eV)...")
            
        # Loop over all bands
        for b in range(n_bands):
            # Get band energies at all k-points
            band_energies = bands[:, b]  # Shape: (N_k,)
            
            # Get eigenvectors for this band at crystal momentum k
            evec = evecs[:, :, b]  # Shape: (N_k, dim)
            
            # Matrix element: phi_tilde(q) * a(k), with phi at full q, evecs at reduced k
            S = phi_at_q[3] * (evec[:, 0] + evec[:, 1])
            Z = phi_at_q[0] * (evec[:, 4] + evec[:, 7])
            X = (phi_at_q[1] + phi_at_q[2]) / np.sqrt(2) * (evec[:, 2] + evec[:, 5])
            Y = -1j * (phi_at_q[1] - phi_at_q[2]) / np.sqrt(2) * (evec[:, 3] + evec[:, 6])
            
            MME = kdp_at_q * (S + X + Y + Z)
            scootch = 0
            
            # Add s* orbital contributions if included
            if self.Simulation_setup.simulation_input["s_star"] == True:
                Ss = phi_at_q[4] * (evec[:, 8] + evec[:, 9])
                MME += kdp_at_q * Ss
                scootch = 2
                
                if self.Simulation_setup.simulation_input["spin_orbit"] == True:
                    Ss_v = phi_at_q[4] * (evec[:, 18] + evec[:, 19])
                    MME += kdp_at_q * Ss_v
            
            # Add spin-orbit split bands if included
            if self.Simulation_setup.simulation_input["spin_orbit"] == True:
                S_v = phi_at_q[3] * (evec[:, 8 + scootch] + evec[:, 9 + scootch])
                Z_v = phi_at_q[0] * (evec[:, 12 + scootch] + evec[:, 15 + scootch])
                X_v = (phi_at_q[1] + phi_at_q[2]) / np.sqrt(2) * (evec[:, 10 + scootch] + evec[:, 13 + scootch])
                Y_v = -1j * (phi_at_q[1] - phi_at_q[2]) / np.sqrt(2) * (evec[:, 11 + scootch] + evec[:, 14 + scootch])
                
                MME += kdp_at_q * (S_v + X_v + Y_v + Z_v)
            
            # Calculate intensity for this band at all k-points
            band_intensity = torch.abs(MME)**2  # Shape: (N_k,)
            
            # Apply Gaussian broadening: distribute intensity to nearby energies
            for ie, E in enumerate(Erange):
                energy_diff = band_energies - E
                weights = torch.exp(-energy_diff**2 / (2 * sigma_E**2))
                
                # Add weighted intensity to this energy slice
                intensity_2d[ie, :] += (band_intensity * weights).to(torch.float64)
        
        # Normalize
        max_intensity = torch.max(intensity_2d)
        if max_intensity > 0:
            intensity_2d /= max_intensity
        
        return intensity_2d, Erange, k_vecs

    def plot_intensity_path_bands(self, vmax=1, vmin=1e-27, figsize=(10, 6), cmap_name='custom'):
        """Plot 2D ARPES intensity map I(k, E)."""
        
        # Calculate intensity
        print("Calculating ARPES intensity map...")
        intensity_2d, energy_array, k_distance = self.intensity_path
        lattice_constant = self.Material.material_parameters_AU["lattice_constant"]
        energy_array *= 27.211386245
        
        # Create figure with black background
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Set up colormap
        if cmap_name == 'custom':
            colors = ["black", "red", "yellow"]
            cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        else:
            cmap = plt.get_cmap(cmap_name)
        
        # Convert to numpy for plotting
        intensity_np = intensity_2d.numpy()
        k_modulus = np.sqrt(np.sum(k_distance.numpy()**2, axis=1)) * lattice_constant / (2*np.pi)
        
        # Determine vmax if not provided
        if vmax is None:
            vmax = intensity_np.max()
        
        # Print diagnostics
        print(f"Intensity range: [{intensity_np.min():.2e}, {intensity_np.max():.2e}]")
        print(f"Using vmin={vmin:.2e}, vmax={vmax:.2e}")
        
        # Plot as image with proper normalization
        im = ax.imshow(intensity_np,
                    extent=[k_modulus[0], k_modulus[-1], 
                            energy_array[0], energy_array[-1]],
                    aspect='auto',
                    origin='lower',
                    cmap=cmap,
                    norm=LogNorm(vmin=vmin, vmax=vmax),  # Now LogNorm is properly imported
                    interpolation='gaussian')
        
        # Colorbar with white text
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('ARPES Intensity', fontsize=12, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Labels with white text
        pol = self.Simulation_setup.simulation_input.get('polarization', 'unknown')
        ax.set_xlabel('k ($2\pi$/a)', fontsize=14, color='white')
        ax.set_ylabel('Energy (eV)', fontsize=14, color='white')
        ax.set_title(f'ARPES Intensity Map (pol: {pol})', fontsize=14, color='white')
        ax.set_xticks(np.arange(9,18,1))
        
        # Set tick colors to white
        ax.tick_params(axis='x', colors='white', labelsize=12)
        ax.tick_params(axis='y', colors='white', labelsize=12)
        
        # Set spine colors to white
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        
        plt.tight_layout()
        
        
        return fig, ax