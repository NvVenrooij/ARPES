# Tight binding ARPES for semiconductors

A Python package for simulating angle-resolved photoemission spectroscopy (ARPES) intensity patterns from semiconductor bulk crystals and delta-layer heterostructures. The simulation combines an empirical sp3s\* tight-binding band structure (with optional spin–orbit coupling and second-nearest-neighbour hopping) with analytic photoemission matrix elements built from hydrogenic atomic orbital Fourier transforms.

The code was developed to support the interpretation of soft- and hard-X-ray ARPES experiments on delta-doped silicon, but the framework is general enough to handle any zinc-blende or diamond-structure semiconductor for which Slater–Koster parameters are available.

## What the simulation does

Given a material, a plane (or path) in three-dimensional crystal momentum space, a photon energy, and a light polarisation, the code produces:

1. **A tight-binding band structure** — eigenvalues and eigenvectors of the sp3s\* Hamiltonian evaluated on the requested k-manifold. With spin–orbit coupling enabled the basis is doubled to 20 spin-orbitals; without it the basis has 10 orbitals per primitive cell (two atoms × {s, p_x, p_y, p_z, s\*}).
2. **A photoemission matrix element** ⟨f | ε · p̂ | i ⟩ evaluated at every k-point in the manifold. The initial state is the tight-binding eigenstate; the final state is approximated as a plane wave at the photoelectron momentum q = k + G, with the sum over reciprocal lattice vectors G handled explicitly. Each tight-binding orbital contributes its analytic Fourier transform φ̃(q), evaluated at the full photoelectron momentum (not the reduced crystal momentum) — — this is the origin of the cross-zone intensity decay in the model.
3. **A 2D isosurface map** — the integrated intensity in a narrow energy window around a chosen binding energy, plotted across a user-defined plane in k-space. This is the simulation analogue of an ARPES constant-energy map.
4. **A 1D band plot along a high-symmetry path** L–Γ–X–U–Γ for diagnostic comparison with literature band structures.

For delta-layer (2DEG) calculations the code additionally multiplies the matrix element by a perpendicular-momentum **envelope function** representing the broken translational symmetry along the confinement direction. Two envelopes are implemented: an analytically derived **finite-square-well** envelope (solved transcendentally for the bound-state wavevector and decay length) and a **piecewise Lorentzian** envelope for the limiting case of strong confinement.

## Physics summary

The matrix element is computed in the dipole approximation with a plane-wave final state,

M(q) = (ε · q) Σ_α e^{−i G · τ_α} Σ_μ c_αμ(k, n) φ̃_αμ(q),   q = k + G,

where τ_α runs over the two atoms of the diamond / zinc-blende basis, μ over the orbitals on each
atom, c_αμ(k, n) are the tight-binding eigenvector components of band n, and φ̃_αμ(q) is the
analytic Fourier transform of orbital μ, evaluated at the full photoelectron momentum q. Three
intensity effects follow from this expression:

- Inter-zone alternation from the two-atom structure factor Σ_α e^{−i G · τ_α}, weighted by the
  band- and orbital-dependent sublattice amplitudes;
- Cross-zone decay from the atomic form factor |φ̃(q)|, which falls off with |q|;
- Intra-zone modulations from momentum-dependent orbital phases in c_αμ(k, n).

For delta layers, the out-of-plane crystal momentum is fixed at the conduction-band minimum ±k₀
and the bulk k_⊥ selection rule is replaced by the Fourier transform of a finite-square-well
envelope, F̃(q_⊥ ∓ k₀ − G_⊥). Reciprocal-lattice vectors G run over the full 3D (bcc) reciprocal
lattice with G_∥ = q_∥ − k_∥.

Summation: intensities are summed incoherently over bands, photoelectron spin, in-plane
reciprocal-lattice vectors (distinct initial states) and the ±k₀ valleys; contributions from
different G_⊥ belonging to the same state are summed coherently. Energies are Gaussian-broadened.

Atomic orbitals: hydrogenic 3s, 3p and 4s (for s*) radial functions with orbital-specific
effective nuclear charges for Si: Z_3s = 4.90 and Z_3p = 4.29 (Clementi–Raimondi), Z_s* = 8.7.
The sign of each form factor follows (−1)^(n−l−1), i.e. positive outer lobes, consistent with
the Slater–Koster convention of Vogl et al. These Z values are specific to Si; other materials
require their own values.

## Features

- **sp3s\* empirical tight-binding** in the Vogl / Klimeck conventions, with optional second-nearest-neighbour (110) hoppings.
- **Spin–orbit coupling** via on-site λ·L·S contributions on each sublattice, doubling the basis from 10 to 20.
- **Photoemission matrix elements** with hydrogenic 3s, 3p_{x,y,z}, and 4s orbital form factors, fully vectorised over k.
- **Bulk and delta-layer modes** in one framework: the same sp3s\* Hamiltonian is reused; the heterostructure simply adds a confinement envelope to the matrix element.
- **Light polarisation** — linear (x, y, z) and right/left circular in the three coordinate planes (kx_ky, kx_kz, ky_kz and their conjugates).
- **Constant-energy plane maps** in arbitrary planes through k-space, defined by two intersection points and an offset along the perpendicular axis.
- **Band structure along the standard L–Γ–X–U–Γ path** with band minima annotated automatically.
- **Materials database** (`materials_db.json`): Si (three parameter sets: full, valence-band-fitted Si_vb, conduction-band-fitted Si_cb), Ge, GaAs (Vogl and Klimeck parametrisations), GaP (Vogl and Klimeck).
- **GPU-friendly tensor backend** — all heavy linear algebra runs through PyTorch, so the simulation transparently uses CUDA if available.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- SciPy (used for the transcendental finite-square-well root finder)

## Repository layout

```
ARPES/
├── main.py               # Example entry point — set parameters, run, plot
├── materials.py          # Material class: loads JSON, unit conversion, lattice + reciprocal-lattice generation
├── simulation_setup.py   # Simulation class: builds k-planes, k-paths, and intersecting reciprocal lattice
├── tight_binding.py      # TightBinding class: Slater–Koster matrix construction (sp3, sp3s*, sp3s*+NNN), diagonalisation, band plotting
├── spin_orbit.py         # On-site spin–orbit Hamiltonian blocks
├── arpes.py              # ARPES class: orbital form factors, matrix element, envelopes, intensity maps
├── classes.py            # Shared / legacy class definitions
├── materials_db.json     # Slater–Koster parameters for Si, Ge, GaAs, GaP variants
└── unit_conversions.json # Physical constants and conversion factors
```

## Usage

The simplest entry point is `main.py`, which sets up a Si calculation, plots the bulk band structure along L–Γ–X–U–Γ, and computes one or two constant-energy maps in user-chosen planes. The skeleton is:

```python
from materials import Material
from simulation_setup import Simulation
from tight_binding import TightBinding
from arpes import ARPES

material = Material("Si")
mps = material.material_parameters_AU

sim_params = {
    'is_plane': True,
    'energy': 1.2,           # binding energy of the slice [eV below the reference]
    'bandwidth': 0.01,       # energy window [eV]
    'v_max': 0.3,            # colour-scale ceiling on the normalised intensity
    'k_ranges': (-1.75, 1.75, 9, 14.5, 0),   # (kx_min, kx_max, kz_min, kz_max, ky_offset) in units of 2π/a
    'intersection_points': [mps['high_symmetry_points']['Xx'],
                            mps['high_symmetry_points']['Xz'],
                            'XxXz'],
    'plane_density': 200,    # grid resolution; must be divisible by 10
    'polarization': 'kx_ky', # right-circular in the xy plane
    'grid': True,
    's_star': True,
    'spin_orbit': True,
    'nnn': False,
    'delta_layer': False,
    'kappa': 0,
    'kz_shift': 0,
    'delta_potential': 0,
    'excited_state?': False,
    'potential well depth': None,
    'potential well width': None,
}

sim = Simulation(material, sim_params)
tb  = TightBinding(material, sim)
arp = ARPES(material, tb, sim)
arp.plot_intensity_plane()
```

To run a delta-layer calculation set `delta_layer = True` and provide a confinement scale (`kappa`, in inverse Bohr radii, of order the lattice constant) together with a finite-well depth in eV and width in lattice constants:

```python
delta_layer = True
kappa = 20                  # decay length ~ lattice constant
potential_well_depth = 0.3  # eV
potential_well_width = 3    # in units of the lattice constant
kz_shift = 0.74             # bound-state wavevector inside the well [2π/a]
```

The envelope is then applied automatically inside `arpes.ARPES.Intensity_plane`.

## Parameter reference

Quantities passed in the `simulation_input` dictionary:

| Key | Meaning |
| --- | --- |
| `is_plane` | `True` for a 2D constant-energy map, `False` for a 1D band plot along L–Γ–X–U–Γ. |
| `energy`, `bandwidth` | Centre and half-width of the energy window [eV] selecting which bands contribute at each k. |
| `v_max` | Upper colour-scale limit for the normalised intensity plot. |
| `k_ranges` | Tuple `(kx_min, kx_max, kz_min, kz_max, ky_offset)` defining the plane extents, in units of 2π/a. |
| `intersection_points` | Two high-symmetry points (in 2π/a units) that span the plane, plus a string label used in plot titles. |
| `plane_density` | Linear resolution of the k-plane grid; must be a multiple of 10. |
| `polarization` | One of `x`, `y`, `z`, `kx_ky`, `kx_kz`, `ky_kz`, `ky_kx`, `kz_kx`, `kz_ky` — linear or right/left circular polarisations. |
| `s_star` | `True` to include the s\* orbital (sp3s\* model); `False` for plain sp3. |
| `spin_orbit` | `True` to include on-site spin–orbit coupling, doubling the basis. |
| `nnn` | `True` to include second-nearest-neighbour (110) hopping terms. |
| `delta_layer` | `True` to apply a confinement envelope along kz. |
| `kappa` | Inverse decay length [Bohr⁻¹] for the confined wavefunction. |
| `kz_shift` | Wavevector offset of the bound-state envelope [2π/a]. |
| `potential well depth`, `potential well width` | Finite-square-well parameters for the analytic envelope, in eV and lattice constants. |
| `excited_state?` | Use the first excited state of the well (odd parity) instead of the ground state. |
| `grid` | Overlay the reciprocal-lattice grid on the intensity map. |

The materials database fields `e_sa`, `e_sc`, `e_pa`, `e_pc`, `e_ssa`, `e_ssc` are the on-site energies; the `v_*` fields are the two-centre Slater–Koster integrals (nearest-neighbour); the `v_*_110` and `v_*_011` fields are the corresponding second-nearest-neighbour parameters; `lambda_so_*` are the on-site spin–orbit coupling constants for anion and cation. All energies are stored in eV in the JSON and converted to Hartree internally; all lengths are stored in nm and converted to Bohr.

## Outputs

`ARPES.plot_intensity_plane()` produces a 2D heatmap of the (normalised) photoemission intensity in the chosen plane, with an overlaid reciprocal-lattice grid when `grid=True`. The intensity is normalised to its maximum on the plane; `v_max` controls how aggressively that maximum is clipped, which is useful when one Brillouin zone dominates and you want to see structure inside the others.

`TightBinding.plot_bands_along_path()` produces the band structure along L–Γ–X–U–Γ, with each band's minimum marked and annotated with energy and k-coordinate.

## Notes and caveats

- The diamond / zinc-blende basis has two atoms; the `Intensity_plane` routine includes the sublattice phase factors explicitly so that the inter-zone alternation pattern from the structure factor is correctly reproduced. This is important when the tight-binding Hamiltonian is evaluated at constant k_z, where these phase components would otherwise be missed.
- Effective nuclear charges for the hydrogenic orbital Fourier transforms are estimated from the nearest-neighbour distance via Z\_eff = 2/d_NN. This is a coarse approximation; for quantitative comparison to experiment it is worth treating Z\_eff as a fitting parameter.
- The reciprocal-lattice intersection routine in `simulation_setup.py` currently uses a dot-product perpendicularity test, which is sensitive to numerical noise in the plane normal and can under-include umklapp vectors near the boundary; a distance-based (radial cutoff) or slab-based selection is more robust if you extend the code.
- All inputs in the materials database are in conventional units (eV, nm); conversions to atomic units (Hartree, Bohr) happen once at construction in `Material.convert_to_atomic_units_and_calculate_Z_eff`.

## Citation

If you use this code, please cite the accompanying paper (reference to be added on publication)
and the archived software release via its Zenodo DOI (see "Cite this repository" on GitHub).


## License

MIT