import matplotlib.pyplot as plt
from simulation_setup import Simulation
from tight_binding import TightBinding
from arpes import ARPES

def build_simulation(material, sim_params):
    sim = Simulation(material, sim_params)
    tb = TightBinding(material, sim)
    arpes = ARPES(material, tb, sim)
    return sim, tb, arpes

def make_base_sim_params(
    *,
    is_plane,
    energy,
    bandwidth,
    k_ranges,
    plane_density,
    polarization,
    grid,
    spin_orbit,
    delta_layer,
    kz_shift,
    delta_potential,
    potential_well_depth,
    potential_well_width,
    v_max=None,
    intersection_points=None,
    ):
    
    return {
        "is_plane": is_plane,
        "energy": energy,
        "bandwidth": bandwidth,
        "v_max": v_max,
        "k_ranges": k_ranges,
        "intersection_points": intersection_points,
        "plane_density": plane_density,
        "polarization": polarization,
        "grid": grid,
        "spin_orbit": spin_orbit,
        "delta_layer": delta_layer,
        "kz_shift": kz_shift,
        "delta_potential": delta_potential,
        "potential well depth": potential_well_depth,
        "potential well width": potential_well_width,
    }


def run_plane(material, base_params, point_a, point_b, plane_name):
    sim_params = dict(base_params)
    sim_params["intersection_points"] = [point_a, point_b, plane_name]

    sim, tb, arpes = build_simulation(material, sim_params)

    plt.figure()
    arpes.plot_intensity_plane()
    plt.show()

    return sim, tb, arpes


def run_path(material, path_params):
    sim, tb, arpes = build_simulation(material, path_params)

    plt.figure()
    tb.plot_bands_along_path()

    plt.figure()
    arpes.plot_intensity_path_bands(vmax=path_params.get("v_max", 1))

    plt.show()
    return sim, tb, arpes
