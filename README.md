# Alpha Processing — GMM Decomposition of Solar Orbiter PAS Ion VDFs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18902395.svg)](https://doi.org/10.5281/zenodo.18902395)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

Automated Gaussian Mixture Model pipeline that decomposes Solar Orbiter PAS (Proton-Alpha Sensor) 3D ion velocity distribution functions into their constituent populations — **proton core, proton beam, and alpha particles** — recovering physical parameters (density, velocity, temperature ∥/⟂) for each.

## Table of Contents

- [Science Goal](#science-goal)
- [Pipeline Overview](#pipeline-overview)
- [Quick Start](#quick-start)
- [Pipeline Steps in Detail](#pipeline-steps-in-detail)
- [How the GMM Decomposition Works](#how-the-gmm-decomposition-works)
- [PAS Instrument Effects](#pas-instrument-effects)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Output Structure](#output-structure)
- [Dependencies](#dependencies)
- [Quality Diagnostics](#quality-diagnostics)
- [Authors](#authors)

---

## Science Goal

Solar Orbiter's PAS instrument measures 3D ion velocity distribution functions (VDFs) on a grid of **11 azimuth × 9 elevation × 96 energy bins**. Each VDF contains overlapping contributions from:

- **Proton core** — the bulk solar wind population
- **Proton beam** — a faster proton population streaming along the magnetic field
- **Alpha particles** — He²⁺ ions with a different mass/charge ratio

This pipeline automatically separates these three populations using a Gaussian Mixture Model.

---

## Pipeline Overview

```
                  ┌─────────────────────┐
  For each day:   │ 1. Download SOAR    │  sunpy_soar_download.py
                  │    data (CDF files) │
                  └────────┬────────────┘
                           │
                  ┌────────▼────────────┐
                  │ 2. GMM fitting      │  gmm_auto_parallelised.py
                  │    (parallelised)   │  → 20 workers default
                  └────────┬────────────┘
                           │
                  ┌────────▼────────────┐
                  │ 3. Save sparse VDFs │  Save_vdfs.py
                  │    to HDF5          │
                  └────────┬────────────┘
                           │
                  ┌────────▼────────────┐
                  │ 4. Cleanup raw +    │  Delete_files.py (Optional, by setting _DELETE)
                  │    intermediate data│
                  └────────┬────────────┘
                           │
                  ┌────────▼────────────┐
                  │ 5. Calculate        │  Cal_Moments_From_VDF.py
                  │    Moments          │
                  └─────────────────────┘
```

A **pre-fetch strategy** downloads the next day's data in the background while the current day's GMM runs, so no wall-clock time is wasted waiting for downloads.

---

## Quick Start

```bash
# 1. Edit pipeline_controller.py to set your date range and parameters
#    Look for the "USER SETTINGS" block (~line 207):
#      PIPELINE_TSTART = datetime(2023, 6, 4, 0, 0, 0)
#      PIPELINE_TEND   = datetime(2023, 6, 5, 23, 59, 59)
#      DT_WANTED   = 4.0
#      N_PROCESSES = 20
#      _DELETE     = True   # False → keep raw data & intermediates

# 2. Run
python pipeline_controller.py
```

That's it. The pipeline will:
1. Download PAS + MAG data from SOAR for each day
2. Fit GMMs to every 4-second time slice
3. Save the separated proton & alpha VDFs as sparse HDF5 files
4. Clean up raw and intermediate files to save disk space
5. If the user wants to have moments, please edit the StartDate and EndDate in Cal_Moments_From_VDFs.py, and run it. 

---

## Pipeline Steps in Detail

### Step 1: Data Download (`sunpy_soar_download.py`)

Downloads four data products per day from the [Solar Orbiter Archive (SOAR)](https://soar.esac.esa.int/soar/):

| Product | Level | Content |
|---------|-------|---------|
| `swa-pas-grnd-mom` | L2 | Ground-calculated moments (density, velocity) |
| `swa-pas-vdf` | L2 | 3D velocity distribution functions |
| `swa-pas-3d` | L1 | Raw counts (for noise estimation) |
| `mag-srf-normal` | L2 | Magnetic field vectors in SRF |

### Step 2: GMM Fitting (`gmm_auto_parallelised.py`)

For each day, the script:

1. **Collects valid time indices** — matches VDF and MAG epochs, filters irregular cadences
2. **Resamples** to the target cadence (default: 4 s)
3. **Calculates one-particle noise level** — 99.99th percentile of `VDF / COUNTS` over the day
4. **Auto-detects initial GMM means** for each time slice using the thermal-safety bulk-velocity method
5. **Fits all 4 GMM covariance types** (`full`, `diag`, `tied`, `spherical`) per slice
6. **Selects the best covariance** via `Flag_Vperp = |Vα,⊥ − Vp,⊥| / |Vp,‖|`
7. **Saves** separated proton (core + beam) and alpha VDFs as pickle files

Processing is embarrassingly parallel across time slices using `ProcessPoolExecutor`.

### Step 3: Save Sparse VDFs (`Save_vdfs.py`)

Converts the per-slice pickle files into **sparse HDF5** format — only non-zero VDF bins are stored — dramatically reducing file size. Each HDF5 file contains:

- `time`: array of int64 nanosecond timestamps
- `value`, `i`, `j`, `k`: sparse VDF data (value at grid index)
- `ptr`: pointer array for fast slicing by time index

### Step 4: Cleanup (`Delete_files.py`)

Deletes the raw CDF data directory and intermediate pickle result directory for each day, freeing disk space. One-particle noise level files are preserved in `result/SO/Noise_Level/`. Handles NFS silly-rename stubs by killing any processes still holding file handles.

### Step 5: Calculate moments (`Cal_Moments_From_VDFs.py`, Manual)

Set the StartDate and EndDate of `Cal_Moments_From_VDFs.py`, and you can calculate the moment of the period that you wanted. Results will be saved as `.csv` files in `results/SO/Moments/`.

## In summay, `pipeline_controller.py` runs step 1-4, and if the user wants moments, please manually run step 5.

---

## How the GMM Decomposition Works

### The Challenge

We have a 3D VDF on a discrete instrument grid. It's a sum of at least three overlapping Maxwellian-like populations. We need to figure out which part of the VDF belongs to which population.

### Particle Resampling Approach

The core insight: phase-space density `f(v)` is a *density* — it tells you how many particles live per unit volume in velocity space. To resample "particles" that follow this density, each PAS grid cell $(i,j,k)$ gets a number of samples proportional to its **phase-space content**:

$$ N_{ijk} = N_{\text{total}} \cdot \frac{f_{ijk} \cdot \Delta V_{ijk}}{\sum f \cdot \Delta V} $$

where the phase-space volume element on the PAS spherical grid is:

$$ \Delta V = v^2 \, \Delta v \, \cos\theta \, \Delta\theta \, \Delta\phi $$

From the resampled point cloud we:

1. **Fit a GMM** in pure 3D velocity space $[V_\parallel, V_{\perp1}, V_{\perp2}]$
2. **Apply posteriors back** to partition the original VDF grid: $f_k(\mathbf{v}) = p_k(\mathbf{v}) \cdot f(\mathbf{v})$, which conserves total phase-space density exactly
3. **Sort components** into [alpha, beam, core] order for downstream moment calculation

This is physically sound: the GMM sees the actual velocity distribution, not a distorted feature space.

### Three Resampling Strategies

| Strategy | How particles are placed within each bin |
|----------|------------------------------------------|
| **centre** | All at the bin-centre velocity (fastest) |
| **uniform** | Uniformly scattered in (v, θ, φ) within the bin volume |
| **interp** | Probabilistically placed where `f` is highest, via trilinear interpolation (slowest, while physically sensible) |

The pipeline uses **interp** by default.
However, accoding to our simulation, **uniform** and **interp** do not give significantly different results in most of the cases.

### Auto-Initialisation

GMM needs decent initial guesses. The pipeline auto-detects them for every time slice:

1. **Proton core**: peak of the 3D VDF (protons dominate by density → argmax works)
2. **Proton beam**: 1× Alfvén speed along **B** (physically motivated)
3. **Alpha particles**: Compute the bulk velocity of all VDF signal above a dividing boundary placed at `Vp + 3×dv_hwhm + 2×VA`. Beyond this boundary, proton phase-space density has dropped to near-zero — whatever remains is alpha-dominated.

No manual tuning, no peak hunting — it works automatically for every slice.

### Covariance Selection

All four sklearn covariance types are tried for each slice. The best is selected using:

$$ \text{Flag}_{V\perp} = \frac{|V_{\alpha,\perp} - V_{p,\perp}|}{|V_{p,\parallel}|} $$

This measures perpendicular coupling: alphas and protons should stream together across **B**. The covariance type giving the smallest `Flag_Vperp` (subject to `f_alpha_max / f_proton_max ≤ 0.2` to reject unphysical fits) wins.

---

## PAS Instrument Effects

Two systematic effects arise because PAS measures energy-per-charge (E/q) assuming proton mass:

### E/q Shift (Apparent Velocity)
An alpha particle (m = 4m_p, q = 2e) at the same *speed* as a proton has twice the E/q. When PAS converts to speed assuming proton mass, **alphas appear √2× faster** than they really are:
$$ v_{\alpha}^{\text{apparent}} = v_{\alpha}^{\text{true}} \times \sqrt{2} $$

### Phase-Space Density Compression
The Jacobian of the E/q → v conversion compresses alpha PSD by factor 4:
$$ f_{\text{measured}} = f_{\text{true}} / 4 $$

### Corrections Applied
- **GMM fitting**: done in proton-equivalent velocity space (no correction needed — we fit what PAS sees)
- **Moment integration**: alpha VDFs are divided by √2 in velocity and multiplied by 4 in PSD before integration, recovering true physical units
- **FAC transform**: VDFs are rotated from spacecraft frame into **field-aligned coordinates** (V_para ∥ **B**, V_perp1, V_perp2 ⟂ **B**)

---

## Configuration

### Pipeline Controller (`pipeline_controller.py`)

Edit the `USER SETTINGS` block:

```python
PIPELINE_TSTART = datetime(2023, 6, 4, 0, 0, 0)   # start of date range
PIPELINE_TEND   = datetime(2023, 6, 5, 23, 59, 59)  # end of date range
DT_WANTED   = 4.0    # desired output cadence (s)
N_PROCESSES = 20     # parallel workers for GMM fitting
_DELETE     = True   # False → keep raw data & intermediate products
```

### Individual Scripts

Each sub-script has its own `# === CONFIG ===` block. The pipeline controller edits these blocks programmatically — you normally don't need to touch them. But you can run scripts standalone:

```bash
# Download a single day
python sunpy_soar_download.py
# (edit TIME_START, TIME_END in the CONFIG block first)

# Run GMM for a single day
python gmm_auto_parallelised.py
# (edit YYMMDD, T_START_ISO, T_END_ISO, etc.)
```

---

## File Structure

```
Alpha_Processing/
├── pipeline_controller.py         # Master orchestrator
├── sunpy_soar_download.py         # SOAR data download
├── gmm_auto_parallelised.py       # GMM fitting (parallelised)
├── Save_vdfs.py                   # Convert pickles → sparse HDF5
├── Delete_files.py                # Cleanup raw + intermediate data
├── Cal_Moments_From_VDFs.py       # Calculate moments
│
├── gmm_resample.py                # Core GMM engine (resampling + fitting)
├── Funcs.py                       # Physics utilities (FAC, overlap, moments I/O)
├── SolarWindPack.py               # SolarWindParticle class + moment integration
├── Constrain_Funcs.py             # Quality diagnostics (overlap, connectedness, smoothness)
│
├── data/SO/{YYYYMMDD}/            # Downloaded CDF files (temporary)
├── result/SO/{YYYYMMDD}/          # Intermediate results (temporary)
│   ├── one_particle_noise_level.npz
│   └── Particles/Ions_auto_resample/{HHMMSS}/
│       ├── Protons.pkl
│       ├── Alphas.pkl
│       └── Final_result.png
├── result/SO/VDFs/                # Final sparse HDF5 output
│   ├── Proton_vdf_{YYYYMMDD}.h5
│   └── Alpha_vdf_{YYYYMMDD}.h5
└── result/SO/Noise_Level/         # Preserved noise level files
    └── one_particle_noise_level_{YYYYMMDD}.npz
```

---

## Output Structure

### Sparse HDF5 VDF Files

Each `.h5` file contains:

| Dataset | Dtype | Shape | Description |
|---------|-------|-------|-------------|
| `time` | int64 | (N_t,) | Unix nanosecond timestamps |
| `value` | float64 | (N_nz,) | Non-zero VDF values |
| `i` | uint8 | (N_nz,) | Azimuth index (0–10) |
| `j` | uint8 | (N_nz,) | Elevation index (0–8) |
| `k` | uint8 | (N_nz,) | Energy index (0–95) |
| `ptr` | int64 | (N_t+1,) | Pointer array: VDF at time `t` is `value[ptr[t]:ptr[t+1]]` |

To load a VDF at time index `t`:
```python
import h5py, numpy as np
from Funcs import load_vdf

time, vdf = load_vdf("result/SO/VDFs/Proton_vdf_20230604.h5", index=0)
# vdf.shape → (11, 9, 96)
# time → pandas Timestamp
```

### Pickle Files (Intermediate)

Each pickle contains a `SolarWindParticle` object with:
- `.species` — `'proton'` or `'alpha'`
- `.time` — measurement datetime
- `.magfield` — **B** vector in SRF (nT)
- `.get_vdf()` — 3D VDF with PAS instrument corrections already applied

---

## Dependencies

```
python >= 3.9
numpy, scipy, matplotlib, pandas
scikit-learn
astropy
sunpy + sunpy-soar
spacepy
h5py
```

All available in the `research_env` conda environment.

---

## Quality Diagnostics

`Constrain_Funcs.py` provides three families of quality checks that can be applied to GMM output:

### 1. Overlap (`cal_overlap_features`)
Measures how much the proton and alpha 1D VDF profiles overlap:
- `overlap_alpha_density` — fraction of alpha intensity under the proton profile
- `overlap_proton_density` — fraction of proton intensity under the alpha profile

Higher overlap → harder separation → larger expected errors.

### 2. Moment Plausibility (`cal_moment_plausibility_features`)
Scores the physical plausibility of recovered moments using soft thresholds on density ratio, alpha–proton drift, temperature ratios, and anisotropy — catches absurd fits.

---



## Authors

- **Hao Ran** — UCL / MSSL (hao.ran.24@ucl.ac.uk)

Created: 2026.07

---

## License

This code is part of ongoing research at UCL/MSSL. Please contact the authors before reuse.
