# SWA-Data-Analysis

Tools for analysing ion measurements from the **Solar Wind Analyzer Proton and Alpha-Particle Sensor (SWA/PAS)** instrument on **Solar Orbiter**.
This repository implements a physics-informed **Gaussian Mixture Model (GMM)** pipeline for separating **proton core**, **proton beam**, and **alpha-particle** populations from 3D velocity distribution functions (VDFs), together with the code to generate a *collared* VDF based on the measurements.

If you find this repository useful, please consider giving it a ⭐.

---

# Getting Started

### Step 0: Download your data into data/SO/yymmdd (e.g., data/SO/20220302)

Data includes:
-swa-pas-3d, -swa-pas-vdf, -swa-pas-grnd-mom, -mag_srf

---

### Step 1: Go to GMM_Hao_Tutorial.ipynb
Inside the notebook, set  

	  timeslice

Plot the VDF and visually determine the manual separation index:

	  dividing_idx

Run the code, it will save the results in a folder called (result/SO/yymmdd/hhmmss).

---
### Choose between the following two options.

### Step 2.1 : Go to GMM_3components.py
In the *main* function, set:

	tstart
	tend

Here, tstart should be the *timeslice* in GMM_Hao_Tutorial.ipynb + 4s. Make sure that the folder names match.

### Step 2.2 : Go to GMM_3components_parallelised.py
**!!! This code is still under developing!!!**
Use this if you are happy to sacrifice accuracy and want to increase the speed of calculating.

  Concept:
  To enable parallelisation while keeping temporal continuity, we process the time series in blocks (e.g. N = 15 slices ≈ 1 min at 4 s cadence).
  1. First timeslice is obtained via Hao_GMM_Tutorial.ipynb
  2. The next block of measurements, all use the results from Hao_GMM_Tutorial.ipynb as initial values.
  3. Take the average of this block's separation, and use it as initial value for the next block.
  4. Thus, blocks depend sequentially on each other, but slices within each block can be processed independently across multiple cores.

Set:

	  tstart
	  tend
	  block_length
	  n_processes
---

### Step 3 (Optional): Get time-series data plots.
We have offered some codes to easily get the timeseries data out of the GMM-separated results. The users can also write their own.

Get_tseries.py: Read the moments and concatenate them in a *.csv* file.
Separation_plot.py: Plot the time_series and the separation results for each timeslice.
Make_movie.py: Take the figures plotted by *Separation_plot.py*, and make a movie.

---

### Step 4 (Optional): Generate a bi-Maxwellian *collared* VDF out of measurements
Please use generate_ALPS_input_collared.py

We get all the measured points from PAS, and interpolate them and merge the interpolation into a bi-Maxwellian distribution background.
The background bi-Maxwellian distribution is called a "collar".
Every step is carefully recorded, and the code is well documented.

---

  
