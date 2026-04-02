#%%

from materials import Material
from run_helpers import *

def main():
    material = Material("Si")
    mps = material.material_parameters_AU

    kmin_y = -3
    kmax_y = 3
    kmin_z = 9
    kmax_z = 14.5

    k_ranges = (kmin_y, kmax_y, kmin_z - 1, kmax_z + 3, 0)
    plane_density = 200

    polarization = "kz_ky"
    delta_layer = False
    grid = True
    spin_orbit = True

    energy = 1.2
    bandwidth = 0.01
    v_max = 0.1

    run_path_calc = False
    run_xx_xz = True
    run_k_xz = True

    if delta_layer:
        kappa = 20
        potential_well_depth = 0.3
        potential_well_width = 3
        kz_shift = -0.74
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
    )

    if run_path_calc:
        run_path(material, path_params)

    if run_xx_xz:
        run_plane(
            material,
            plane_base_params,
            mps["high_symmetry_points"]["Xx"],
            mps["high_symmetry_points"]["Xz"],
            "XxXz",
        )

    if run_k_xz:
        run_plane(
            material,
            plane_base_params,
            mps["high_symmetry_points"]["K"],
            mps["high_symmetry_points"]["Xz"],
            "KXz",
        )


if __name__ == "__main__":
    main()
