# AE 370 – Wing Vibration (Euler–Bernoulli Beam)

This repository contains a finite-difference, method-of-lines
solver for the transverse vibration of a clamped–free wing
modeled as an Euler–Bernoulli beam.

## Contents

- `src/beam_solver.py`  
  Finite-difference spatial discretization of the fourth-order
  Euler–Bernoulli beam equation with clamped–free boundary
  conditions and RK4 time integration.

- `src/mode_analysis.py`  
  Eigenvalue and mode-shape analysis of the discretized beam.

- `src/convergence_tests.py`  
  Temporal and spatial convergence studies used to justify
  time-step and grid-size choices.

- `plots/`  
  All figures included in the final report.

## Reproducibility

All figures in the report can be reproduced by running the
scripts in the `src/` directory with the parameter values
described in the report.

python src/beam_solver.py

python src/mode_analysis.py

python src/convergence_tests.py

