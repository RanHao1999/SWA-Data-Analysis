# SWA-Data-Analysis

Tools for analysing ion measurements from the **Solar Orbiter Solar Wind Analyzer (SWA/PAS)** instrument.  
This repository implements a physics-informed **Gaussian Mixture Model (GMM)** pipeline for separating **proton core**, **proton beam**, and **alpha-particle** populations from 3D velocity distribution functions (VDFs), together with utilities for generating ALPS inputs, plotting diagnostics, and producing movies.

If you find this repository useful, please consider giving it a ⭐.

---

# 🛰️ Overview

SWA/PAS measures 3D ion VDFs every 4 seconds by sweeping through energy, elevation, and azimuth.  
This repository provides:

- Accurate **GMM-based separation** of ions (core / beam / α)
- **Parallelised GMM** with block-wise warm-start for high efficiency (have to sacrifice some accuracy though)
- Tools for **time-series extraction**, **diagnostic plotting**, and **movie generation**
- A **collar-generation module** to prepare ALPS-ready inputs
  
This codebase is actively used for scientific studies involving Solar Orbiter data, kinetic instabilities, and ion-scale physics.

---

# 🚀 Getting Started

This repository assumes you have Solar Orbiter SWA/PAS data stored in a structure like:
- data/

   -SO/

      -yymmdd/

         -swa-pas-3d

         -swa-pas-vdf

         -swa-pas-grnd-mom

         -mag_srf



# 🎯 Workflow Summary

The recommended workflow is:

1. **Check the first timeslice manually using the tutorial notebook**  
2. **Run the full GMM separation** (sequential or parallel)
3. **Extract time-series moments**
4. **Plot separation results**
5. **Optionally generate ALPS input**
6. **Optionally create a movie**

Details below.

---
# 🔍 1. Separating the first Timeslice (GMM_Hao_Tutorial.ipynb)

Inside the notebook, set  
  timeslice
Plot the VDF and visually determine the manual separation index:
  dividing_idx

---
# GMM Separation, of the following 2, choose one that you favour.
 ## 🧪 2.1. Running GMM on Long Intervals (Sequential)
 Use: GMM_3components.py
 Set:
   tstart (the timeslice (GMM_Hao_Tutorial.ipynb) + 4s, just make sure that the folder names match)
   tend

## 🧪 2.2 Parallel GMM.
  To accelerate the computation, use: GMM_3component_parallelised.py
  
  Concept:
  To enable parallelisation while keeping temporal continuity, we process the time series in blocks (e.g. N = 15 slices ≈ 1 min at 4 s cadence).
   1. First block:
        - Fit the first slice sequentially with a generic initial guess.
        - Use its converged parameters as the common initial guess for all
          other slices in the block, which are then fitted in parallel.

   2. Subsequent blocks:
        - Take the average GMM solution from the previous block.
        - Use it as the initial guess for the entire new block.
        - Fit all slices in the block in parallel.

Thus, blocks depend sequentially on each other, but slices within each block can be processed independently across multiple cores.
Set:
  tstart
  tend
  block_length
  n_processes
  








  
