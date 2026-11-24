import matplotlib.pyplot as plt
from simulation_setup import Simulation
from tight_binding import TightBinding
from arpes import ARPES
from pathlib import Path
import json
import numpy as np

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
    save_path=None,
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
        "save_path": save_path,
    }


def run_plane(material, base_params, point_a, point_b, plane_name):
    sim_params = dict(base_params)
    sim_params["intersection_points"] = [point_a, point_b, plane_name]

    sim, tb, arpes = build_simulation(material, sim_params)

    fig = plt.figure()
    arpes.plot_intensity_plane()
    save_fig(fig, sim_params, plane_name)      # label = "XxXz" / "KXz"
    save_data({"intensity_raw": arpes.intensity_raw,
               "k_manifold": sim.k_manifold}, 
               sim_params, 
               plane_name
            )

    plt.show()
    return sim, tb, arpes


def run_path(material, path_params):
    sim, tb, arpes = build_simulation(material, path_params)

    fig_bands = plt.figure()
    tb.plot_bands_along_path()

    fig_int, _ = arpes.plot_intensity_path_bands(vmax=path_params.get("v_max"))

    save_fig(fig_bands, path_params, "bands")
    save_fig(fig_int,   path_params, "path_intensity")
    save_data({"intensity_raw": arpes.intensity_raw,
               "Erange_Ha": arpes.Erange,
               "k_path": arpes.k_path}, path_params, "path_intensity")

    plt.show()
    return sim, tb, arpes


def fig_name(params, label):
    """Build a descriptive filename stem from the simulation parameters."""
    parts = [
        label,
        f"pol{params['polarization']}" if params.get("polarization") else None,
        f"E{params['energy']:.2f}" if params.get("energy") is not None else None,
        f"kz{params['kz_shift']:.2f}",
        "delta" if params["delta_layer"] else "bulk",
        "soc" if params["spin_orbit"] else "nosoc",
    ]
    return "_".join(p for p in parts if p)


def save_fig(fig, params, label):
    """Save fig into params['save_path'] under an auto-generated name. No-op if None."""
    if params.get("save_path") is None:
        return None
    outdir = Path(params["save_path"])
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{fig_name(params, label)}.pdf"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    print(f"wrote {path}")
    return path

def save_data(arrays, params, label):
    """Save raw plot arrays + parameters as .npz next to the figure. No-op if save_path is None."""
    if params.get("save_path") is None:
        return None
    outdir = Path(params["save_path"])
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{fig_name(params, label)}.npz"
    np.savez_compressed(
        path,
        **{k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)) for k, v in arrays.items()},
        params=json.dumps(params, default=str),
    )
    print(f"wrote {path}")
    return path

def save_cb(arpes, I_plus, I_min, ExtendedRange=None):
    params = arpes.Simulation_setup.simulation_input
    label  = f"{params['intersection_points'][-1]}_CB"      # e.g. "XxXz_CB"

    if ExtendedRange is not None:
        label += "_ER"

    I_tot = I_plus + I_min
    arpes.intensity_plane = I_tot / I_tot.max()

    fig = plt.figure()
    arpes.plot_intensity_plane()

    save_fig(fig, params, label)
    save_data({"intensity_raw": I_tot}, params, label)
    plt.show()