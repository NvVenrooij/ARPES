# ARPES Simulation

Simulates angle-resolved photoemission spectroscopy (ARPES) intensity patterns for semiconductor heterostructures using an sp3s* tight-binding model with spin-orbit coupling.

## Features

- Tight-binding band structure with sp3s* basis and spin-orbit coupling
- Photoemission matrix elements ⟨f|ε·p̂|i⟩ with atomic orbital Fourier transforms
- Bulk crystals (Si, Ge, GaAs, GaP) and delta-layer quantum well structures
- 2D isosurface k-space maps and 1D band structure along k-paths
- Configurable light polarisation and energy broadening
- Material parameters loaded from JSON database


## Requirements

Python 3.8+ with PyTorch, NumPy, Matplotlib, SciPy.