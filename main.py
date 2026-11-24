#%%
import torch
import numpy as np
from materials import Material
from run_helpers import *

def main():
    material = Material("Si")
    mps = material.material_parameters_AU

    kmin_x = -4 
    kmax_x = 4  
    kmin_y = -3
    kmax_y = 3
    kmin_z = 9 
    kmax_z = 17

    k_ranges = (kmin_x, kmax_x, kmin_z, kmax_z, 0)
    plane_density = 500

    polarization = "kz_ky"
    delta_layer = True
    grid = True
    spin_orbit = True

    energy = 1.2
    bandwidth = 0.01
    v_max = 1

    run_path_calc = False
    run_xx_xz = True
    run_k_xz = True

    path = '/Users/nielsvanvenrooij/Library/CloudStorage/Dropbox/Mac/Documents/phD/Projects/ARPES/Code/Figures'

    if delta_layer:
        kappa = 20
        potential_well_depth = 0.3
        potential_well_width = 3
        kz_shift = 0.74
        delta_potential = 0.0

    else:
        kappa = 0
        potential_well_depth = None
        potential_well_width = None
        kz_shift = 0.0
        delta_potential = 0.0

    plane_base_params = make_base_sim_params(
        is_plane=True,
        energy=energy,
        bandwidth=bandwidth,
        v_max=v_max,
        k_ranges=k_ranges,
        plane_density=plane_density,
        polarization=polarization,
        grid=grid,
        spin_orbit=spin_orbit,
        delta_layer=delta_layer,
        kz_shift=kz_shift,
        delta_potential=delta_potential,
        potential_well_depth=potential_well_depth,
        potential_well_width=potential_well_width,
        save_path= path,
    )

    plane_base_params["kappa"] = kappa

    path_params = make_base_sim_params(
        is_plane=False,
        energy=None,
        bandwidth=bandwidth,
        v_max=v_max,
        k_ranges=k_ranges,
        plane_density=plane_density,
        polarization=polarization,
        grid=None,
        spin_orbit=spin_orbit,
        delta_layer=delta_layer,
        kz_shift=kz_shift,
        delta_potential=delta_potential,
        potential_well_depth=potential_well_depth,
        potential_well_width=potential_well_width,
        save_path= path,
    )

    if run_path_calc:
        sim_path, tb_path, arpes_path = run_path(material, path_params)

    if run_xx_xz:
        sim_xxz, tb_xxz, arpes_xxz = run_plane(
            material,
            plane_base_params,
            mps["high_symmetry_points"]["Xx"],
            mps["high_symmetry_points"]["Xz"],
            "XxXz",
        )

    if run_k_xz:
        sim_kxz, tb_kxz, arpes_kxz = run_plane(
            material,
            plane_base_params,
            mps["high_symmetry_points"]["K"],
            mps["high_symmetry_points"]["Xz"],
            "KXz",
        )

    
    # end of main()
    out = {}
    if run_path_calc:
        out["path"] = (sim_path, tb_path, arpes_path)
    if run_xx_xz:
        out["XxXz"] = (sim_xxz, tb_xxz, arpes_xxz)
    if run_k_xz:
        out["KXz"] = (sim_kxz, tb_kxz, arpes_kxz)
    return out


if __name__ == "__main__":
    results = main()