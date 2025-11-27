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

---

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
   ```python
   timeslice #The beginning of the interval that you are interested in.
   # Plot the VDF and visually determine the manual separation index
   dividing_idx

---

 # 🧪 2. Running GMM on Long Intervals (Sequential)
