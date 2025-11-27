# SWA-Data-Analysis

Tools for analysing ion measurements from the **Solar Orbiter Solar Wind Analyzer (SWA/PAS)** instrument.  
This repository implements a physics-informed **Gaussian Mixture Model (GMM)** pipeline for separating **proton core**, **proton beam**, and **alpha-particle** populations from 3D velocity distribution functions (VDFs), together with utilities for generating ALPS inputs, plotting diagnostics, and producing movies.

If you find this repository useful, please consider giving it a ⭐.

---

# 🛰️ Overview

SWA/PAS measures 3D ion VDFs every 4 seconds by sweeping through energy, elevation, and azimuth.  
This repository provides:

- Accurate **GMM-based separation** of ions (core / beam / α)
- **Parallelised GMM** with block-wise warm-start for high efficiency
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

