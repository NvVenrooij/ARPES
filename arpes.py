import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from matplotlib.colors import LinearSegmentedColormap, LogNorm, PowerNorm


class ARPES:
    def __init__(self, Material, Tight_binding, Simulation_setup):
        self.Material = Material
        self.Tight_binding = Tight_binding
        self.Simulation_setup = Simulation_setup

        if self.Simulation_setup.simulation_input.get('is_plane', False):
            if self.Simulation_setup.simulation_input.get('delta_layer', True):
                out = self.Intensity_plane_delta_layer()
            else:
                out = self.Intensity_plane_bulk()
            (self.intensity_plane,   # normalised intensity, (Ni, Nj)
             self.intensity_raw,     # unnormalised intensity; add these when combining runs
             self.MME,               # amplitudes per band and spin, (nb, 2, Ni, Nj)
             self.MME_weights,       # Gaussian energy weight per state, (nb, Ni, Nj)
             ) = out
        else:
            (self.intensity_path,    # normalised I(E, k), (n_E, N_k)
             self.intensity_raw,     # unnormalised I(E, k)
             self.MME,               # amplitudes per band and spin, (nb, 2, N_k)
             self.Erange,            # energy axis in Hartree, (n_E,)
             self.k_path,            # k-points, (N_k, 3)
             ) = self.Intensity_path_bands()

    def Phi310(self, k_vecs):

        """Calculates the values of the Fourier transformed 3pz-orbital.
        
            input:
                - k_vecs: a 3xN or 3xNxN tensor containing all k-space vectors on the 
                path
                
            output: 
                - a 3xN or 3xNxN tensor containing all values for the
                Fourier transform at every k-vector in k_vecs on the path"""

        #Z = self.Material.material_parameters_AU['effective_nuclear_charge']
        Z = 4.29 #5.63

        k_mag = torch.sqrt(torch.sum(k_vecs**2, dim=-1))
        R_over_k = 144*np.sqrt(6)*Z**3.5*(9*k_mag**2 - Z**2)/(Z**2 + 9*k_mag**2)**4   # radial integral / k
        return 4*np.pi*1J*0.5*np.sqrt(3/np.pi)*k_vecs[..., 2]*R_over_k

    def Phi311(self, k_vecs):
        
        """Calculates the values of the Fourier transformed 3px-orbital
        
            input:
                - k_vecs: a 3xN or 3xNxN tensor containing all k-space vectors on the 
                path

            output: 
                - a 3xN or 3xNxN tensor containing all values for the
                Fourier transform at every k-vector in k_vecs on the path"""

        #Z = self.Material.material_parameters_AU['effective_nuclear_charge']
        Z = 4.29 #5.63

        k_mag = torch.sqrt(torch.sum(k_vecs**2, dim=-1))
        R_over_k = 144*np.sqrt(6)*Z**3.5*(9*k_mag**2 - Z**2)/(Z**2 + 9*k_mag**2)**4   # radial integral / k
        return 4*np.pi*1J*0.5*np.sqrt(3/(2*np.pi))*(k_vecs[..., 0] + k_vecs[..., 1]*1J)*R_over_k

    def Phi31m1(self, k_vecs):
        
        """Calculates the values of the Fourier transformed 3py-orbital.
        
            input:
                - k_vecs: a 3xN or 3xNxN tensor containing all k-space vectors on the 
                path

            output: 
                - a 3xN or 3xNxN tensor containing all values for the
                Fourier transform at every k-vector in k_vecs on the path"""

        #Z = self.Material.material_parameters_AU['effective_nuclear_charge']
        Z = 4.29 #5.63

        k_mag = torch.sqrt(torch.sum(k_vecs**2, dim=-1))
        R_over_k = 144*np.sqrt(6)*Z**3.5*(9*k_mag**2 - Z**2)/(Z**2 + 9*k_mag**2)**4   # radial integral / k
        return 4*np.pi*1J*0.5*np.sqrt(3/(2*np.pi))*(k_vecs[..., 0] - k_vecs[..., 1]*1J)*R_over_k

    def Phi300(self, k_vecs):

        """Calculates the Fourier transform of the 3s-orbital.
        
            input:
                - k_vecs: a 3xN or 3xNxN tensor containing all k-space vectors on the 
                path

            output: 
                - a 3xN or 3xNxN tensor containing all values for the
                Fourier transform at every k-vector in k_vecs on the path"""
        
        #Z = self.Material.material_parameters_AU['effective_nuclear_charge']
        Z = 4.9 #6.08

        k2 = torch.sum(k_vecs**2, dim=-1)
        I = 36*np.sqrt(3)*Z**2.5*(3*k2 - Z**2)*(27*k2 - Z**2)/(Z**2 + 9*k2)**4
        return 4*np.pi*1/(2*np.sqrt(np.pi))*I

    def Phi400(self, k_vecs):

        """Calculates the Fourier transform of the 4s(s*) -orbital.
        
            input:
                - k_vecs: a 3xN or 3xNxN tensor containing all k-space vectors on the 
                path

            output: 
                - a 3xN or 3xNxN tensor containing all values for the
                Fourier transform at every k-vector in k_vecs on the path"""
        
        #Z = self.Material.material_parameters_AU['effective_nuclear_charge']
        Z = 8.7 #10.8

        k2 = torch.sum(k_vecs**2, dim=-1)
        I = 128*Z**2.5*(16*k2 - Z**2)*(256*k2**2 - 96*Z**2*k2 + Z**4)/(Z**2 + 16*k2)**5
        return -4*np.pi*1/(2*np.sqrt(np.pi))*I


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
        M_up = kdpm * (S + X + Y + Z + Ss)

        if not use_soc:
            return M_up, None
        
        S_v  = phiS  * (WA * evec[:, idx_soc["A_s_dn"]]     + WB * evec[:, idx_soc["B_s_dn"]])
        X_v  = Xfac  * (WA * evec[:, idx_soc["A_px_dn"]]    + WB * evec[:, idx_soc["B_px_dn"]])
        Y_v  = Yfac  * (WA * evec[:, idx_soc["A_py_dn"]]    + WB * evec[:, idx_soc["B_py_dn"]])
        Z_v  = phiZ  * (WA * evec[:, idx_soc["A_pz_dn"]]    + WB * evec[:, idx_soc["B_pz_dn"]])
        Ss_v = phiSs * (WA * evec[:, idx_soc["A_sstar_dn"]] + WB * evec[:, idx_soc["B_sstar_dn"]])
        M_dn = kdpm * (S_v + X_v + Y_v + Z_v + Ss_v)
        return M_up, M_dn
  
    
    def Intensity_plane_bulk(self):
        '''Bulk ARPES intensity in a k-plane with Gaussian energy broadening.

        I(q) = sum_b w_b(q) * sum_sigma |M_{b,sigma}(q)|^2,
        M_{b,sigma}(q) = (eps . q) * sum_alpha c_{alpha,sigma}^b(k) phi_tilde_alpha(q).
        Bands and spins are summed incoherently.

        Returns
        -------
        intensity     : (Ni, Nj) normalised to its maximum
        intensity_raw : (Ni, Nj) unnormalised
        MME           : (nb, 2, Ni, Nj) complex amplitudes per band and spin
        weights_all   : (nb, Ni, Nj) Gaussian energy weight per state
        '''
        k_vecs = self.Simulation_setup.k_manifold      # full photoelectron momenta q
        bands  = self.Tight_binding.bands
        evecs  = self.Tight_binding.evecs

        Ni, Nj, n_bands = bands.shape
        E       = self.Simulation_setup.simulation_input["energy"]    / 27.2113862459
        sigma_E = self.Simulation_setup.simulation_input["bandwidth"] / 27.2113862459

        phi_funcs = [self.Phi310, self.Phi311, self.Phi31m1, self.Phi300, self.Phi400]
        phi_at_q  = [phi_func(k_vecs) for phi_func in phi_funcs]
        kdp_at_q  = self.k_dot_lambda(k_vecs)

        MME         = torch.zeros((n_bands, 2, Ni, Nj), dtype=torch.complex128)
        weights_all = torch.zeros((n_bands, Ni, Nj), dtype=torch.float64)

        print(f"Calculating intensity plane with Gaussian broadening (σ = {sigma_E*27.211:.3f} eV)...")

        for b in range(n_bands):
            weights = torch.exp(-(bands[:, :, b] - E)**2 / (2 * sigma_E**2))   # (Ni, Nj)
            significant_mask = weights > 1e-4
            if not torch.any(significant_mask):
                continue

            Mi, Mj = torch.where(significant_mask)
            evec = evecs[Mi, Mj, :, b]
            ones = torch.ones(len(Mi), dtype=torch.complex128)

            M_up, M_dn = self.orbital_matrix_element(
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

            MME[b, 0, Mi, Mj] = M_up
            if M_dn is not None:
                MME[b, 1, Mi, Mj] = M_dn
            weights_all[b, Mi, Mj] = weights[Mi, Mj].to(torch.float64)

        # incoherent over spin (dim 1) and bands (dim 0)
        intensity_raw = (weights_all * (MME.abs()**2).sum(dim=1)).sum(dim=0)
        mx = intensity_raw.max()
        intensity = intensity_raw / mx if mx > 0 else intensity_raw

        return intensity, intensity_raw, MME, weights_all
        

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

        nb   = 20 if use_soc else 10
        Macc = torch.zeros((nb, 2, Ni * Nj), dtype=torch.complex128, device=device)  # coherent over G_z only
        Wacc = torch.zeros((nb, Ni * Nj), dtype=torch.float64, device=device)        # energy weight per state

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

                M_up, M_dn = self.orbital_matrix_element(
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

                Macc[b, 0, idx[ids]] += M_up
                if M_dn is not None:
                    Macc[b, 1, idx[ids]] += M_dn
                Wacc[b, idx[ids]] = w[ids]

        MME     = Macc.reshape(nb, 2, Ni, Nj)       # amplitude per band b, spin σ (coherent over G_z)
        weights = Wacc.reshape(nb, Ni, Nj)          # Gaussian energy weight of each state
        intensity_raw = (weights * (MME.abs()**2).sum(dim=1)).sum(dim=0)
        intensity = intensity_raw / intensity_raw.max() if intensity_raw.max() > 0 else intensity_raw
        return intensity, intensity_raw, MME, weights

    def Intensity_plane_delta_layer(
        self,
        keep_gy_strips=1,
        gy_tol=1e-8,
        w_mask_min=1e-2,
        energy_mask_min=1e-4,
        incoherent_gpar=True,                                   # NEW: False reproduces the old result exactly
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

        # NEW: sort G so that all G with the same (Gx, Gy) come one after another
        order = sorted(range(G_list.shape[0]),
                       key=lambda n: (round(G_list[n, 0].item() / measure),
                                      round(G_list[n, 1].item() / measure)))
        G_list = G_list[order]

        # NEW: sanity check -- the speed-up needs q_par to depend only on the column (true for XxXz, KXz)
        if not torch.allclose(q_plane[..., :2], q_plane[0:1, :, :2].expand(Ni, Nj, 2)):
            raise ValueError("Speed-up assumes the cut contains the z-axis (q_par depends only on the column).")

        nb   = 20 if use_soc else 10
        Macc = torch.zeros((nb, 2, Ni * Nj), dtype=torch.complex128, device=device)  # coherent over G_z only
        Wacc = torch.zeros((nb, Ni * Nj), dtype=torch.float64, device=device)        # energy weight per state

        eig_cache   = {}                                                             # NEW
        I_acc       = torch.zeros(Ni * Nj, dtype=torch.float64, device=device)       # NEW
        Macc_all    = torch.zeros_like(Macc)                                         # NEW
        Wacc_all    = torch.zeros_like(Wacc)                                         # NEW
        current_key = None                                                           # NEW

        for G in G_list:
            Gx, Gy, Gz = G

            # NEW: (Gx, Gy) label; kz is overwritten by sp3s, so eigenstates depend only on this and the column
            key = (int(round(Gx.item() / measure)), int(round(Gy.item() / measure)))

            # NEW: when (Gx, Gy) changes, square-and-add the finished group, then start a fresh one
            if incoherent_gpar and current_key is not None and key != current_key:
                I_acc += (Wacc * (Macc.abs()**2).sum(dim=1)).sum(dim=0)
                Macc_all += Macc
                Wacc_all = torch.maximum(Wacc_all, Wacc)
                Macc.zero_()
                Wacc.zero_()
            current_key = key

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

            # NEW: replaces the per-pixel sp3s + eigh. One diagonalisation per column per (Gx, Gy),
            # then look up the column of each selected pixel. Gives identical evals/evecs.
            if key not in eig_cache:
                k_cols = q_plane[0] - G[None, :]          # one k-point per column
                H_cols = self.Tight_binding.sp3s(spin_orbit=use_soc, k_manifold=k_cols)
                eig_cache[key] = torch.linalg.eigh(H_cols)
            evals_c, evecs_c = eig_cache[key]
            cols  = idx % Nj                              # column of each selected pixel
            evals = evals_c[cols]
            evecs = evecs_c[cols]

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

                M_up, M_dn = self.orbital_matrix_element(
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

                Macc[b, 0, idx[ids]] += M_up
                if M_dn is not None:
                    Macc[b, 1, idx[ids]] += M_dn
                Wacc[b, idx[ids]] = w[ids]

        # NEW: square-and-add the last group
        I_acc += (Wacc * (Macc.abs()**2).sum(dim=1)).sum(dim=0)
        Macc_all += Macc
        Wacc_all = torch.maximum(Wacc_all, Wacc)

        MME     = Macc_all.reshape(nb, 2, Ni, Nj)   # amplitude per band b, spin σ (summed over all G)
        weights = Wacc_all.reshape(nb, Ni, Nj)      # Gaussian energy weight of each state
        intensity_raw = I_acc.reshape(Ni, Nj)       # NEW: was computed from MME and weights
        intensity = intensity_raw / intensity_raw.max() if intensity_raw.max() > 0 else intensity_raw
        return intensity, intensity_raw, MME, weights
    
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


    def plot_intensity_plane(self, intensity_array=None, gamma=0.4):

        k_vecs = self.Simulation_setup.k_manifold
        lattice_constant = self.Material.material_parameters_AU["lattice_constant"]
        if intensity_array is None:
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
        norm = PowerNorm(gamma=gamma, vmin=0, vmax=1)        # same gamma for every panel; intensity is normalised to max = 1
        # Display the intensity array using imshow
        plt.imshow(
            intensity_array,
            extent=[scaling*kmin_x, scaling*kmax_x, kmin_z, kmax_z],
            origin='lower',
            norm=norm,                                       # replaces vmax=v_max
            cmap=cmap,
            interpolation='gaussian'
        )

        # Get current axes
        ax = plt.gca()
        # Get current axes
        ax = plt.gca()

        # Set axis labels with larger font size and bold font weight
        ax.set_xlabel('kx ($2\pi$/a)', fontsize=14)
        ax.set_ylabel('kz ($2\pi$/a)', fontsize=14)

        # Increase the font size of tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)

    
    def Intensity_path_bands(self):
        """ARPES intensity along a k-path as function of energy, Gaussian broadened.

        I(E, k) = sum_b exp(-(E_b(k) - E)^2 / 2 sigma^2) * sum_sigma |M_{b,sigma}(q)|^2.
        Bands and spins are summed incoherently.

        Returns
        -------
        intensity_2d  : (n_E, N_k) normalised to its maximum
        intensity_raw : (n_E, N_k) unnormalised
        MME           : (nb, 2, N_k) complex amplitudes per band and spin
        Erange        : (n_E,) energies in Hartree (numpy)
        k_vecs        : (N_k, 3) k-points
        """
        k_vecs = self.Simulation_setup.k_manifold      # full photoelectron momenta q
        bands  = self.Tight_binding.bands              # (N_k, n_bands)
        evecs  = self.Tight_binding.evecs              # (N_k, dim, n_bands)

        N_k, n_bands = bands.shape

        Emax = 0.1  / 27.2113862459
        Emin = -6.2 / 27.2113862459
        step = (Emax - Emin) / N_k
        Erange   = np.arange(Emin, Emax, step)
        Erange_t = torch.as_tensor(Erange, dtype=torch.float64)

        sigma_E = self.Simulation_setup.simulation_input["bandwidth"] / 27.2113862459

        phi_funcs = [self.Phi310, self.Phi311, self.Phi31m1, self.Phi300, self.Phi400]
        phi_at_q  = [phi_func(k_vecs) for phi_func in phi_funcs]
        kdp_at_q  = self.k_dot_lambda(k_vecs)
        ones      = torch.ones(N_k, dtype=torch.complex128, device=k_vecs.device)

        MME           = torch.zeros((n_bands, 2, N_k), dtype=torch.complex128)
        intensity_raw = torch.zeros((len(Erange), N_k), dtype=torch.float64)

        print(f"Calculating intensity map with Gaussian broadening (σ = {sigma_E*27.211:.3f} eV)...")

        for b in range(n_bands):
            band_energies = bands[:, b].to(torch.float64)          # (N_k,)
            evec = evecs[:, :, b]                                   # (N_k, dim)

            M_up, M_dn = self.orbital_matrix_element(
                evec=evec,
                phiS=phi_at_q[3], phiZ=phi_at_q[0],
                phiX1=phi_at_q[1], phiX2=phi_at_q[2], phiSs=phi_at_q[4],
                WA=ones, WB=ones, kdpm=kdp_at_q,
            )

            MME[b, 0] = M_up
            band_intensity = M_up.abs()**2
            if M_dn is not None:
                MME[b, 1] = M_dn
                band_intensity = band_intensity + M_dn.abs()**2    # incoherent over spin

            weights = torch.exp(-(band_energies[None, :] - Erange_t[:, None])**2 / (2 * sigma_E**2))
            intensity_raw += weights * band_intensity[None, :].to(torch.float64)

        mx = intensity_raw.max()
        intensity_2d = intensity_raw / mx if mx > 0 else intensity_raw

        return intensity_2d, intensity_raw, MME, Erange, k_vecs
        
    def plot_intensity_path_bands(self, vmax=None, scale="log", dynamic_range=1e7, gamma=0.4, figsize=(10, 6), cmap_name='custom'):
        """Plot 2D ARPES intensity map I(k, E)."""
        
        # Calculate intensity
        print("Calculating ARPES intensity map...")
        intensity_2d = self.intensity_path
        energy_array = self.Erange * 27.211386245     # new array; no in-place mutation
        k_distance   = self.k_path
        lattice_constant = self.Material.material_parameters_AU["lattice_constant"]
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
        
        if vmax is None:
            vmax = intensity_np.max()
        if scale == "log":
            norm = LogNorm(vmin=vmax / dynamic_range, vmax=vmax)   # full range, no clipping
        else:
            norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax)

        print(f"Intensity range: [{intensity_np.min():.2e}, {intensity_np.max():.2e}]")
        print(f"Using {scale} scale, vmax={vmax:.2e}")        
        # Print diagnostics
        print(f"Intensity range: [{intensity_np.min():.2e}, {intensity_np.max():.2e}]")
        print(f"Using gamma={gamma}, vmax={vmax:.2e}")
        
        # Plot as image with gamma scale (same gamma as the plane maps)
        im = ax.imshow(intensity_np,
                    extent=[k_modulus[0], k_modulus[-1], 
                            energy_array[0], energy_array[-1]],
                    aspect='auto',
                    origin='lower',
                    cmap=cmap,
                    norm=norm,
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