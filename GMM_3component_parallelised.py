"""
In this code, we use the GMM custering method to separate the PAS observations to Proton Core/ Proton Beam/ Alpha.

Also, we will try all possible combinations, and find the best one in this code.

Idea comes from the paper: https://doi.org/10.1051/0004-6361/202243719 (Rossana. De Marco et al. 2023)

Author: @Hao Ran (hao.ran.24@ucl.ac.uk).

========================================================================================================================
GMM setting:
1. n_components: 3
2. X = [V_para, V_perp1, V_perp2, |V|, vdf]
3. First initial values are determined manually, with VA.
   From the 2nd on, the GMM will use the result of the previous one as a new initial value.
========================================================================================================================

========================================================================================================================
Block-wise warm-start for parallel GMM fitting
-----------------------------------------------------------------------------
To enable parallelisation while keeping temporal continuity, we process the
time series in blocks (e.g. N = 15 slices ≈ 1 min at 4 s cadence).

   1. First block:
        - Fit the first slice sequentially with a generic initial guess.
        - Use its converged parameters as the common initial guess for all
          other slices in the block, which are then fitted in parallel.

   2. Subsequent blocks:
        - Take the average GMM solution from the previous block.
        - Use it as the initial guess for the entire new block.
        - Fit all slices in the block in parallel.

Thus, blocks depend sequentially on each other, but slices within each block
 can be processed independently across multiple cores.
 -----------------------------------------------------------------------------
========================================================================================================================

========================================================================================================================
v1 CHANGES (Hao Ran, June 2026):
-----------------------------------------------------------------------------
Instead of collecting ALL days' data and pre-calculating ALL noise levels
before running any GMM, we now process DAY-BY-DAY:

   For each day:
       1. Check data (collect idx_times for that day)
       2. Calculate one-particle noise level for that day
       3. Run GMM blocks for that day
       4. Move to the next day

This allows downloading data for Day N+1 while Day N is being processed.
The initial_means still propagate across days via warm-start.
========================================================================================================================
"""

# import functions, including the ones that I wrote.
import os
# CRITICAL: Pin BLAS threads to 1 per worker BEFORE numpy/scipy imports.
# Without this, each of the 30 parallel workers spawns 2 OpenBLAS threads,
# causing 60+ threads fighting for 64 cores → load avg spikes to 150.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import time
os.chdir(sys.path[0])

from spacepy import pycdf
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from datetime import datetime, timedelta
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema, savgol_filter
import bisect
import pickle
import gc
import concurrent.futures

from Funcs import *
from SolarWindPack import *

# Set to True for debugging; False for production speed.

def wait_for_data(yymmdd, timeout=3600, poll_interval=30):
    """
    Wait for a day's data directory to appear (e.g. still downloading).
    Polls every `poll_interval` seconds, gives up after `timeout` seconds.

    Returns True if data is ready, False if timeout exceeded.
    """
    data_dir = f"data/SO/{yymmdd}"
    waited = 0
    while not os.path.isdir(data_dir):
        if waited >= timeout:
            print(f"  [WAIT] Timed out after {timeout}s waiting for {yymmdd} data.")
            return False
        print(f"  [WAIT] {yymmdd} data not found, retrying in {poll_interval}s "
              f"(elapsed: {waited}s)...")
        time.sleep(poll_interval)
        waited += poll_interval
    if waited > 0:
        print(f"  [WAIT] {yymmdd} data appeared after {waited}s.")
    return True


def FindIndexinInterval(tstart, tend, epoch):
    # Given the epoch of the data, find all the indexes within the given interval.
    if tstart > tend:
        raise ValueError("tstart should be smaller than tend.")
    result = [[idx, epoch[idx]] for idx in range(len(epoch)) if tstart <= epoch[idx] <= tend]
    return result

def find_irregular_times(tstart, tend, epoch_vdf, expected, tol=0.5):
    """
    Find indices (in original array) where the PAS timestamps within [tstart, tend]
    repeat or are too close (< expected - tol).

    Only timestamps inside [tstart, tend] are considered,
    so results match FindIndexinInterval.
    """

    if tstart > tend:
        raise ValueError("tstart should be smaller than tend.")

    bad_indices = []
    bad_dts = []

    # First collect all indices within the interval
    idx_in_interval = [
        i for i in range(len(epoch_vdf))
        if tstart <= epoch_vdf[i] <= tend
    ]

    # Now look for repeated/too-close times strictly within this set
    lower = expected - tol   # typically 3.1 s

    for k in range(len(idx_in_interval) - 1):

        i = idx_in_interval[k]
        j = idx_in_interval[k+1]

        dt = (epoch_vdf[j] - epoch_vdf[i]).total_seconds()

        if dt < lower:
            bad_indices.append(i)
            bad_dts.append(dt)

    return bad_indices, bad_dts

def days_between(t_start, t_end):
    day = t_start.date()
    end = t_end.date()
    return [(day := day + timedelta(days=1)) and (day - timedelta(days=1)).strftime("%Y%m%d")
            for _ in range((end - day).days + 1)]

def collect_all_idx_times(days, t_start, t_end, dt_seconds):
    """Original function: collect idx_times for ALL days at once.
       Kept for backward compatibility; v1 uses collect_day_idx_times instead."""
    all_idx_times = []

    for yymmdd in days:
        print("\n=== Processing", yymmdd, "===")

        # Load the day's PAS file
        data_list = os.listdir(f"data/SO/{yymmdd}")
        vdf_fname = next(file for file in data_list
                         if 'pas-vdf' in file and not file.startswith('._'))
        vdf_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{vdf_fname}")
        epoch_vdf = vdf_cdffile['Epoch'][...]

        mag_fname = next(file for file in data_list
                         if 'mag-srf' in file and not file.startswith('._'))
        mag_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{mag_fname}")
        epoch_mag = mag_cdffile['EPOCH'][...]

        # Here, make sure t_start and t_end does not include data where MAG has no measurement!
        t_start_thisday = max(t_start, epoch_mag[0])
        t_end_thisday = min(t_end, epoch_mag[-1])

        # Get today's indices
        idx_times = FindIndexinInterval(t_start_thisday, t_end_thisday, epoch_vdf)
        print(len(idx_times), "time indices found in total.")

        # Find irregular times (using your function)
        bad_idx, deltas = find_irregular_times(t_start_thisday, t_end_thisday, epoch_vdf, expected=dt_seconds)
        print("Irregular intervals:", len(bad_idx))

        # Optional: print details
        for idx, dt in zip(bad_idx, deltas):
            print(f"Index {idx}: dt = {dt:.3f} s  ({epoch_vdf[idx]} → {epoch_vdf[idx+1]})")

        # Remove bad indices
        bad_set = set(bad_idx)
        idx_times = [item for item in idx_times if item[0] not in bad_set]
        print(len(idx_times), "indices left after filtering.")

        # Append to the grand list
        all_idx_times.extend(idx_times)

        vdf_cdffile.close()

    return all_idx_times


def collect_day_idx_times(yymmdd, t_start, t_end, dt_seconds):
    """
    v1: Collect idx_times for a SINGLE day.
    This replaces the per-day loop inside collect_all_idx_times so we can
    process each day as soon as its data is available.
    """
    print("\n=== Checking data for", yymmdd, "===")

    # Load the day's PAS file
    data_list = os.listdir(f"data/SO/{yymmdd}")
    vdf_fname = next(file for file in data_list
                     if 'pas-vdf' in file and not file.startswith('._'))
    vdf_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{vdf_fname}")
    epoch_vdf = vdf_cdffile['Epoch'][...]

    mag_fname = next(file for file in data_list
                     if 'mag-srf' in file and not file.startswith('._'))
    mag_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{mag_fname}")
    epoch_mag = mag_cdffile['EPOCH'][...]

    # Make sure t_start and t_end does not include data where MAG has no measurement!
    t_start_thisday = max(t_start, epoch_mag[0])
    t_end_thisday = min(t_end, epoch_mag[-1])

    # Get today's indices
    idx_times = FindIndexinInterval(t_start_thisday, t_end_thisday, epoch_vdf)
    print(len(idx_times), "time indices found in total.")

    # Find irregular times
    bad_idx, deltas = find_irregular_times(t_start_thisday, t_end_thisday, epoch_vdf, expected=dt_seconds)
    print("Irregular intervals:", len(bad_idx))

    # Print details for bad ones
    for idx, dt in zip(bad_idx, deltas):
        print(f"Index {idx}: dt = {dt:.3f} s  ({epoch_vdf[idx]} → {epoch_vdf[idx+1]})")

    # Remove bad indices
    bad_set = set(bad_idx)
    idx_times = [item for item in idx_times if item[0] not in bad_set]
    print(len(idx_times), "indices left after filtering.")

    vdf_cdffile.close()
    mag_cdffile.close()

    return idx_times


def resample_to_idx_time(idx_times, target_interval=4.0):
    if not idx_times:
        return []

    # 1. Always keep the first element
    resampled_list = [idx_times[0]]
    last_kept_time = idx_times[0][1]

    # 2. Iterate through the rest
    for i in range(1, len(idx_times)):
        current_idx, current_time = idx_times[i]

        # Calculate time difference
        time_diff = (current_time - last_kept_time).total_seconds()

        # 3. If the gap is 4s or more, keep it
        if time_diff >= target_interval:
            resampled_list.append([current_idx, current_time])
            last_kept_time = current_time

    return resampled_list


# Functions for plotting.
def log10_vdf(vdf):
    vdf = np.array(vdf)  # Ensure vdf is a NumPy array
    mask = vdf > 0  # Create a mask for positive values
    result = np.zeros_like(vdf)  # Initialize an array of zeros
    result[mask] = np.log10(vdf[mask])  # Compute log10 only for positive values
    return result

def log10_1D_dist(vel, vdf):
    y = log10_vdf(np.sum(vdf, axis=(0, 1)))
    mask = y != 0
    x = vel[mask]
    y = y[mask]
    return x, y

def plot_one(ax1, ax2, vel, vdf_total, f_core, f_beam, f_alpha, co_type):
    x, y = log10_1D_dist(vel, vdf_total)
    ax1.plot(x, y, label='Total', color='black')
    ax1.scatter(x, y, s=20, color='black', marker='s')
    x, y = log10_1D_dist(vel, f_core)
    ax1.plot(x, y, label='Core', color='red')
    ax1.scatter(x, y, s=10, color='red')
    x, y = log10_1D_dist(vel, f_beam)
    ax1.plot(x, y, label='Beam', color='blue')
    ax1.scatter(x, y, s=10, color='blue')
    x, y = log10_1D_dist(vel, f_alpha)
    ax1.plot(x, y, label='Alpha', color='green')
    ax1.scatter(x, y, s=10, color='green')
    ax1.set_ylim(-12, -5)
    ax1.set_title(co_type)
    ax1.set_xlabel('Vel [km/s]')
    ax1.set_ylabel('log10(VDF)')
    ax1.legend()

    x, y = log10_1D_dist(vel, vdf_total)
    ax2.plot(x, y, label='Total', color='black')
    ax2.scatter(x, y, s=20, color='black', marker='s')
    x, y = log10_1D_dist(vel, f_core + f_beam)
    ax2.plot(x, y, label='Proton', color='red')
    ax2.scatter(x, y, s=10, color='red')
    x, y = log10_1D_dist(vel, f_alpha)
    ax2.plot(x, y, label='Alpha', color='green')
    ax2.scatter(x, y, s=10, color='green')
    ax2.set_ylim(-12, -5)
    ax2.set_xlabel('Vel [km/s]')
    ax2.set_ylabel('log10(VDF)')
    ax2.legend()

    return 0


def _prepare_gmm_data(V_para, V_perp1, V_perp2, vdf_corrected, n_component=3):
    """
    Build feature matrix X and compute KMeans initial weights ONCE.
    These are shared across all 4 covariance-type GMM fits, saving
    3 redundant X constructions and 3 KMeans runs per slice.
    """
    non_zero_idx = np.where(vdf_corrected > 0)
    non_zero_vdf = vdf_corrected[non_zero_idx]
    non_zero_vpara = V_para[non_zero_idx]
    non_zero_vperp1 = V_perp1[non_zero_idx]
    non_zero_vperp2 = V_perp2[non_zero_idx]
    non_zero_magni = np.sqrt(non_zero_vpara**2 + non_zero_vperp1**2 + non_zero_vperp2**2)

    X = np.column_stack([non_zero_vpara, non_zero_vperp1, non_zero_vperp2, non_zero_magni, non_zero_vdf])

    # KMeans for weight initialisation (n_init=1 is fine for this purpose)
    kmeans = KMeans(n_clusters=n_component, n_init=1, random_state=0).fit(X)
    labels = kmeans.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    initial_weights = counts / len(labels)

    return X, non_zero_idx, initial_weights


def _cal_GMM_fast(X, non_zero_idx, vdf_corrected, co_type, initial_means, initial_weights, n_component):
    """
    Run a single GMM fit using pre-computed X and KMeans weights.
    Handles component reordering (core/beam/alpha identification).
    """
    gmm_kwargs = {
        "n_components": n_component,
        "random_state": 42,
        "covariance_type": co_type,
        "means_init": initial_means,
        "weights_init": initial_weights,
        "max_iter": 100,
        "tol": 1e-3,
    }

    gmm = GaussianMixture(**gmm_kwargs).fit(X)
    probas = gmm.predict_proba(X)

    f_all = [np.zeros_like(vdf_corrected) for _ in range(n_component)]

    for i in range(n_component):
        f_all[i][non_zero_idx] = probas[:, i] * vdf_corrected[non_zero_idx]

    # Determine alpha: scan from large speed to low speed,
    # whichever component starts at a higher speed is alpha.
    def find_first_nonzero_index(f_1d):
        return np.argmax(f_1d > 0)

    f0_1d = np.sum(f_all[0], axis=(0, 1))
    f1_1d = np.sum(f_all[1], axis=(0, 1))
    f2_1d = np.sum(f_all[2], axis=(0, 1))

    a = find_first_nonzero_index(f0_1d)
    b = find_first_nonzero_index(f1_1d)
    c = find_first_nonzero_index(f2_1d)

    idx = min(a, b, c)
    values_at_idx = [f0_1d[idx], f1_1d[idx], f2_1d[idx]]
    alpha_index = int(np.argmax(values_at_idx))

    # Among remaining two, determine core vs beam by peak value
    remaining_indices = [i for i in range(n_component) if i != alpha_index]
    peak_values = [np.max(f_all[i]) for i in remaining_indices]
    if peak_values[0] >= peak_values[1]:
        core_index = remaining_indices[0]
        beam_index = remaining_indices[1]
    else:
        core_index = remaining_indices[1]
        beam_index = remaining_indices[0]

    # Reorder: [alpha, beam, core] → user-facing order
    sort_indices = [alpha_index, beam_index, core_index]

    f_all_sorted = [f_all[i] for i in sort_indices]
    covariance_sorted = [gmm.covariances_[i] for i in sort_indices]
    means_sorted = [gmm.means_[i] for i in sort_indices]
    weights_sorted = [gmm.weights_[i] for i in sort_indices]

    # Clip tiny values
    for f in f_all_sorted:
        f[f < 1e-14] = 0

    return f_all_sorted, [means_sorted, covariance_sorted, weights_sorted], probas


def cal_GMM(V_para, V_perp1, V_perp2, vdf_corrected, co_type, initial_means, n_component):
    """
    Legacy wrapper — kept for backward compatibility.
    New code should call _prepare_gmm_data once then _cal_GMM_fast for each co_type.
    """
    X, non_zero_idx, initial_weights = _prepare_gmm_data(
        V_para, V_perp1, V_perp2, vdf_corrected, n_component)
    return _cal_GMM_fast(X, non_zero_idx, vdf_corrected, co_type,
                         initial_means, initial_weights, n_component)


def get_initial_means_from_objects(Protons, Alphas):
    magfield = Protons.magfield
    (Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz) = fieldAlignedCoordinates(magfield[0], magfield[1], magfield[2])
    # Get the initial values for GMM.
    u_proton_core = cal_bulk_velocity_Spherical(Protons, component='core')
    u_proton_core_Baligned = rotateVectorIntoFieldAligned(u_proton_core[0], u_proton_core[1], u_proton_core[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    u_proton_beam = cal_bulk_velocity_Spherical(Protons, component='beam')
    u_proton_beam_Baligned = rotateVectorIntoFieldAligned(u_proton_beam[0], u_proton_beam[1], u_proton_beam[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)

    # For alphas, we need to multiply the velocity by sqrt(2) to let GMM separate in the PAS reference.
    u_alpha = cal_bulk_velocity_Spherical(Alphas) * np.sqrt(2)
    u_alpha_Baligned = rotateVectorIntoFieldAligned(u_alpha[0], u_alpha[1], u_alpha[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)

    diff_beam_core = (np.array(u_proton_beam_Baligned) - np.array(u_proton_core_Baligned)) / 1e3
    diff_alpha_core = (np.array(u_alpha_Baligned) - np.array(u_proton_core_Baligned)) / 1e3

    # get VA
    density_p = cal_density_Spherical(Protons) # m^-3
    density_a = cal_density_Spherical(Alphas) # m^-3
    B_magnitude = np.sqrt(np.sum(magfield**2)) * 1e-9  #T
    mu0 = 4 * np.pi * 1e-7 # N/A^2
    mp = 1.67*1e-27 # kg
    ma = 4 * mp
    VA = B_magnitude / np.sqrt(mu0 * (density_p * mp + mu0 * density_a * ma)) / 1000.0 #  km/s

    initial_means = np.array([
        [0, 0, 0, 0, Protons.get_vdf().max()],
        [VA, 0, 0, VA, Protons.get_vdf().max() / 10.0],
        np.append(np.append(diff_alpha_core, np.linalg.norm(diff_alpha_core)), Alphas.get_vdf().max())
    ])

    return initial_means

def remove_noise(vdf_corrected, min_cluster_size=5, connectivity=1, positive_threshold=0.0):
    """
    Remove disconnected positive-VDF clusters smaller than `min_cluster_size`.

    Parameters
    ----------
    vdf_corrected : ndarray
        3D VDF array, e.g. shape (11, 9, 96).
    min_cluster_size : int
        Minimum number of connected positive voxels needed to keep a cluster.
    connectivity : int
        1 -> 6-neighbour, 2 -> 18-neighbour, 3 -> 26-neighbour.
    positive_threshold : float
        Values <= this threshold are treated as zero/no measurement.

    Returns
    -------
    ndarray
        Cleaned VDF array.
    """
    positive_mask = np.isfinite(vdf_corrected) & (vdf_corrected > positive_threshold)

    if not np.any(positive_mask):
        return vdf_corrected.copy()

    structure = scipy.ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
    labels, num_labels = scipy.ndimage.label(positive_mask, structure=structure)

    if num_labels == 0:
        return vdf_corrected.copy()

    label_sizes = np.bincount(labels.ravel(), minlength=num_labels + 1)

    keep_label = np.zeros(num_labels + 1, dtype=bool)
    keep_label[1:] = label_sizes[1:] >= min_cluster_size

    keep_mask = keep_label[labels]

    cleaned_vdf = vdf_corrected.copy()
    cleaned_vdf[~keep_mask] = 0

    return cleaned_vdf


# Module-level cache for day-constant data. Set by preload_day_data() before
# workers are forked — each worker inherits this via copy-on-write.
_day_data = None


def preload_day_data(yymmdd):
    """
    Pre-load day-constant data (Energy/Elevation/Azimuth grids, epochs,
    noise level, filenames) so workers don't re-read them from CDF.
    Sets the module-level _day_data dict.
    """
    global _day_data
    print(f"  Preloading day-constant data for {yymmdd}...")

    data_list = os.listdir(f"data/SO/{yymmdd}")
    vdf_fname = next(file for file in data_list
                     if "pas-vdf" in file and not file.startswith("._"))
    grnd_fname = next(file for file in data_list
                      if "pas-grnd-mom" in file and not file.startswith("._"))
    mag_fname = next(file for file in data_list
                     if "mag-srf-normal" in file and not file.startswith("._"))

    vdf_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{vdf_fname}")
    mag_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{mag_fname}")

    qp = 1.60217662e-19   # C
    mp = 1.6726219e-27    # kg
    vel_raw = np.sqrt(2 * vdf_cdffile["Energy"][...] * qp / mp) / 1e3  # km/s float

    theta_raw = vdf_cdffile["Elevation"][...]  # deg float
    phi_raw = vdf_cdffile["Azimuth"][...]      # deg float

    epoch_vdf = vdf_cdffile["Epoch"][...]
    epoch_mag = mag_cdffile["EPOCH"][...]

    vdf_cdffile.close()
    mag_cdffile.close()

    loaded = np.load(f"result/SO/{yymmdd}/one_particle_noise_level.npz")
    noise_level = loaded["noise_level"]

    _day_data = {
        "yymmdd": yymmdd,
        "vel_raw": vel_raw, "theta_raw": theta_raw, "phi_raw": phi_raw,
        "epoch_vdf": epoch_vdf, "epoch_mag": epoch_mag,
        "noise_level": noise_level,
        "vdf_fname": vdf_fname, "grnd_fname": grnd_fname, "mag_fname": mag_fname,
    }
    print(f"  Day-constant data preloaded ({len(epoch_vdf)} VDF epochs, {len(epoch_mag)} MAG epochs).")


def all_process(idx_time, initial_means):
    """
    A function with all the processes, with figures plotted.
    """
    tsliceindex_vdf = idx_time[0]
    tslice_vdf = idx_time[1]

    yymmdd = tslice_vdf.strftime('%Y%m%d')
    hhmmss = tslice_vdf.strftime('%H%M%S')
    result_path = f'result/SO/{yymmdd}/Particles/Ions/{hhmmss}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    dd = _day_data  # module-level preloaded day-constant data

    if dd is not None and dd["yymmdd"] == yymmdd:
        # --- Fast path: use preloaded grids, only open CDF for per-slice reads ---
        vdf_fname = dd["vdf_fname"]
        grnd_fname = dd["grnd_fname"]
        mag_fname = dd["mag_fname"]
        one_particle_noise_level = dd["noise_level"]
        epoch_vdf = dd["epoch_vdf"]
        epoch_mag = dd["epoch_mag"]

        qp = 1.60217662e-19  # C
        mp = 1.6726219e-27   # kg
        vel = dd["vel_raw"] * (u.km / u.s)
        theta = dd["theta_raw"] * u.deg
        phi = dd["phi_raw"] * u.deg

        # Open CDF only for per-slice reads
        vdf_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{vdf_fname}")
        grnd_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{grnd_fname}")
        mag_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{mag_fname}")
    else:
        # --- Slow path: full CDF read (backward compat / direct calls) ---
        data_list = os.listdir(f"data/SO/{yymmdd}")

        vdf_fname = next(file for file in data_list if "pas-vdf" in file and not file.startswith("._"))
        grnd_fname = next(file for file in data_list if "pas-grnd-mom" in file and not file.startswith("._"))
        mag_fname = next(file for file in data_list if "mag-srf-normal" in file and not file.startswith("._"))

        vdf_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{vdf_fname}")
        grnd_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{grnd_fname}")
        mag_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{mag_fname}")

        loaded_data = np.load(f"result/SO/{yymmdd}/one_particle_noise_level.npz")
        one_particle_noise_level = loaded_data["noise_level"]

        epoch_vdf = vdf_cdffile["Epoch"][...]
        epoch_mag = mag_cdffile["EPOCH"][...]

        qp = 1.60217662e-19  # C
        mp = 1.6726219e-27   # kg
        vel = np.sqrt(2 * vdf_cdffile["Energy"][...] * qp / mp) / 1e3 * (u.km / u.s)
        theta = vdf_cdffile["Elevation"][...] * u.deg
        phi = vdf_cdffile["Azimuth"][...] * u.deg

    # --- Common path: per-slice data reads + processing ---
    # The magnetic field is obtained by calculating the average of the 1-second magnetic field data around the time slice.
    try:
        tslice_vdf_start = epoch_vdf[tsliceindex_vdf] - timedelta(seconds=0.5)
        tslice_vdf_end = epoch_vdf[tsliceindex_vdf] + timedelta(seconds=0.5)
        tsliceindex_mag_start = bisect.bisect_left(epoch_mag, tslice_vdf_start)
        tsliceindex_mag_end = bisect.bisect_left(epoch_mag, tslice_vdf_end)
        magF_SRF = mag_cdffile["B_SRF"][tsliceindex_mag_start:tsliceindex_mag_end].mean(axis=0)
    except Exception:
        # In the cases where, the code goes from one day to the next, it a bit tricky to concatenate the two days' mag data.
        # So, in this case, we just take the closest mag data point.
        idx_mag = bisect.bisect_left(epoch_mag, epoch_vdf[tsliceindex_vdf])
        magF_SRF = mag_cdffile["B_SRF"][idx_mag - 1]

    # --- NaN guard: skip slice if MAG data is missing (data gap) ---
    if not np.all(np.isfinite(magF_SRF)):
        vdf_cdffile.close()
        grnd_cdffile.close()
        mag_cdffile.close()
        return initial_means

    # Read data from SO product.
    V_bulk_SRF = grnd_cdffile["V_SRF"][tsliceindex_vdf]
    vdf = vdf_cdffile["vdf"][tsliceindex_vdf]

    # Calculate the local Alfven speed to set initial means.
    B_magnitude = np.sqrt(np.sum(magF_SRF**2)) * 1e-9  # T
    density = grnd_cdffile["N"][tsliceindex_vdf] * 1e6  # m^-3
    mu0 = 4 * np.pi * 1e-7  # N/A^2
    mp = 1.67 * 1e-27  # kg
    VA = B_magnitude / np.sqrt(mu0 * density * mp) / 1000.0  # km/s

    # Close the cdf files to save memory.
    vdf_cdffile.close()
    grnd_cdffile.close()
    mag_cdffile.close()

    # Any anode with measurement below the noise level is considered as noise, and removed.
    # Let's try not removing an interval first.
    vdf_corrected = vdf.copy()
    vdf_corrected = remove_noise(vdf_corrected)

    # Let's first get the f and (vx, vy, vz) grid points.
    # Vectorized: pre-compute trig with broadcasting instead of triple loop.
    # IMPORTANT: theta/phi from CDF are in degrees; convert to radians for np.cos/sin.
    vel_kms = vel.value  # strip astropy units, all km/s floats from here
    v_bulk = np.asarray(V_bulk_SRF, dtype=float)  # km/s from CDF, already float

    t_rad = np.radians(theta.value)[np.newaxis, :, np.newaxis]  # (1, 9, 1)
    p_rad = np.radians(phi.value)[:, np.newaxis, np.newaxis]    # (11, 1, 1)
    v3d   = vel_kms[np.newaxis, np.newaxis, :]                  # (1, 1, 96)

    cos_t = np.cos(t_rad)
    sin_t = np.sin(t_rad)
    cos_p = np.cos(p_rad)
    sin_p = np.sin(p_rad)

    vx = -v3d * cos_t * cos_p - v_bulk[0]
    vy =  v3d * cos_t * sin_p - v_bulk[1]
    vz = -v3d * sin_t          - v_bulk[2]

    (Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz) = fieldAlignedCoordinates(magF_SRF[0], magF_SRF[1], magF_SRF[2])
    (V_para, V_perp1, V_perp2) = rotateVectorIntoFieldAligned(vx, vy, vz, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    (V_para_bulk, V_perp1_bulk, V_perp2_bulk) = rotateVectorIntoFieldAligned(V_bulk_SRF[0], V_bulk_SRF[1], V_bulk_SRF[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    V_perp = np.sqrt(V_perp1**2 + V_perp2**2)

    def log10_vdf(vdf):
        vdf = np.array(vdf)  # Ensure vdf is a NumPy array
        mask = vdf > 0  # Create a mask for positive values
        result = np.zeros_like(vdf)  # Initialize an array of zeros
        result[mask] = np.log10(vdf[mask])  # Compute log10 only for positive values
        return result

    y = log10_vdf(np.sum(vdf_corrected, axis=(0, 1)))
    mask = y != 0
    x = vel[mask]
    y = y[mask]

    if _PLOT:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y, label='Corrected VDF')
        ax.scatter(x, y, s=20, color='red')
        ax.set_ylim(-12, -5)
        for idx, (xi, yi) in enumerate(zip(x.value, y)):
            ax.annotate(str(np.where(mask)[0][idx]), (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
        ax.set_xlabel('Vel [km/s]')
        ax.set_ylabel('log10(VDF)')
        # plt.savefig(result_path + '/Corrected_VDF_1D.png')
        plt.close()

    # Choose if you want to remove the one-particle noise level.
    vdf_2ndcorrected = vdf_corrected.copy()
    #vdf_2ndcorrected[vdf_2ndcorrected <= one_particle_noise_level] = 0

    # Noise-level subtraction applied below if threshold is set

    # Plot to see the 1D vdf.
    y = log10_vdf(np.sum(vdf_corrected, axis=(0, 1)))
    mask = y != 0
    x = vel[mask]
    y = y[mask]

    #fig, ax = plt.subplots(figsize=(8, 5))
    #ax.plot(x, y, label='Corrected VDF')
    #ax.scatter(x, y, s=20, color='red')
    #ax.set_ylim(-12, -5)
    #for idx, (xi, yi) in enumerate(zip(x.value, y)):
    #    ax.annotate(str(np.where(mask)[0][idx]), (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
    #ax.set_xlabel('Vel [km/s]')
    #ax.set_ylabel('log10(VDF)')
    # plt.savefig(result_path + '/Corrected_VDF_Cleaned.png')
    #plt.close()

    # Initial means
    # Here are two choices, if the beam is really not separate enough, I strongly suggest to use VA as initial input for beam.
    #initial_means = np.array([
    #    [0, 0, 0, 0, Protons_initial.get_vdf(component='core').max()],
    #    np.append(np.append(diff_beam_core, np.linalg.norm(diff_beam_core)), Protons_initial.get_vdf(component='beam').max()),
    #    np.append(np.append(diff_alpha_core, np.linalg.norm(diff_alpha_core)), Alphas_initial.get_vdf().max())
    #])

    n_component = 3
    # Pre-compute X matrix and KMeans weights ONCE, shared across all 4 covariance types
    X, non_zero_idx, gmm_weights = _prepare_gmm_data(
        V_para, V_perp1, V_perp2, vdf_2ndcorrected, n_component)

    # Make sure no data error don't kill the whole pool!
    try:
        f_diag, dist_paras_diag, probas_diag = _cal_GMM_fast(
            X, non_zero_idx, vdf_2ndcorrected, 'diag', initial_means, gmm_weights, n_component)
        f_alpha_diag, f_beam_diag, f_core_diag = f_diag
        f_spherical, dist_paras_spherical, probas_spherical = _cal_GMM_fast(
            X, non_zero_idx, vdf_2ndcorrected, 'spherical', initial_means, gmm_weights, n_component)
        f_alpha_spherical, f_beam_spherical, f_core_spherical = f_spherical
        f_tied, dist_paras_tied, probas_tied = _cal_GMM_fast(
            X, non_zero_idx, vdf_2ndcorrected, 'tied', initial_means, gmm_weights, n_component)
        f_alpha_tied, f_beam_tied, f_core_tied = f_tied
    except Exception as e:
        print(f"[SKIP] {tslice_vdf} GMM failed: {e}")
        # THIS is key to keep block warm-start stable
        return initial_means

    # Carry on with diag, tied, and spherical.
    Protons_diag = SolarWindParticle('proton', time=tslice_vdf, magfield=magF_SRF, grid=[theta.value, phi.value, vel.value * 1e3], coord_type='Spherical')
    Protons_diag.set_vdf(f_core_diag, component='core')
    Protons_diag.set_vdf(f_beam_diag, component='beam')
    Alphas_diag = SolarWindParticle('alpha', time=tslice_vdf, magfield=magF_SRF, grid=[theta.value, phi.value, vel.value * 1e3 / np.sqrt(2)], coord_type='Spherical')
    Alphas_diag.set_vdf(f_alpha_diag * 4)

    Protons_spherical = SolarWindParticle('proton', time=tslice_vdf, magfield=magF_SRF, grid=[theta.value, phi.value, vel.value * 1e3], coord_type='Spherical')
    Protons_spherical.set_vdf(f_core_spherical, component='core')
    Protons_spherical.set_vdf(f_beam_spherical, component='beam')
    Alphas_spherical = SolarWindParticle('alpha', time=tslice_vdf, magfield=magF_SRF, grid=[theta.value, phi.value, vel.value * 1e3 / np.sqrt(2)], coord_type='Spherical')
    Alphas_spherical.set_vdf(f_alpha_spherical * 4)

    Protons_tied = SolarWindParticle('proton', time=tslice_vdf, magfield=magF_SRF, grid=[theta.value, phi.value, vel.value * 1e3], coord_type='Spherical')
    Protons_tied.set_vdf(f_core_tied, component='core')
    Protons_tied.set_vdf(f_beam_tied, component='beam')
    Alphas_tied = SolarWindParticle('alpha', time=tslice_vdf, magfield=magF_SRF, grid=[theta.value, phi.value, vel.value * 1e3 / np.sqrt(2)], coord_type='Spherical')
    Alphas_tied.set_vdf(f_alpha_tied * 4)


    # Print the moments to see what the results look like.
    Vpcore_diag = cal_bulk_velocity_Spherical(Protons_diag, 'core') / 1e3
    Vpbeam_diag = cal_bulk_velocity_Spherical(Protons_diag, 'beam') / 1e3
    Valpha_diag = cal_bulk_velocity_Spherical(Alphas_diag) / 1e3
    Vproton_diag = cal_bulk_velocity_Spherical(Protons_diag) / 1e3
    Vpcore_spherical = cal_bulk_velocity_Spherical(Protons_spherical, 'core') / 1e3
    Vpbeam_spherical = cal_bulk_velocity_Spherical(Protons_spherical, 'beam') / 1e3
    Valpha_spherical = cal_bulk_velocity_Spherical(Alphas_spherical) / 1e3
    Vproton_spherical = cal_bulk_velocity_Spherical(Protons_spherical) / 1e3
    Vpcore_tied = cal_bulk_velocity_Spherical(Protons_tied, 'core') / 1e3
    Vpbeam_tied = cal_bulk_velocity_Spherical(Protons_tied, 'beam') / 1e3
    Valpha_tied = cal_bulk_velocity_Spherical(Alphas_tied) / 1e3
    Vproton_tied = cal_bulk_velocity_Spherical(Protons_tied) / 1e3

    VpcoreDiagBaligned = rotateVectorIntoFieldAligned(Vpcore_diag[0], Vpcore_diag[1], Vpcore_diag[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    VpcoreDiagPara, VpcoreDiagPerp = VpcoreDiagBaligned[0], np.sqrt(VpcoreDiagBaligned[1]**2 + VpcoreDiagBaligned[2]**2)
    VpbeamDiagBaligned = rotateVectorIntoFieldAligned(Vpbeam_diag[0], Vpbeam_diag[1], Vpbeam_diag[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    VpbeamDiagPara, VpbeamDiagPerp = VpbeamDiagBaligned[0], np.sqrt(VpbeamDiagBaligned[1]**2 + VpbeamDiagBaligned[2]**2)
    ValphaDiagBaligned = rotateVectorIntoFieldAligned(Valpha_diag[0], Valpha_diag[1], Valpha_diag[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    ValphaDiagPara, ValphaDiagPerp = ValphaDiagBaligned[0], np.sqrt(ValphaDiagBaligned[1]**2 + ValphaDiagBaligned[2]**2)

    VpcoreSphericalBaligned = rotateVectorIntoFieldAligned(Vpcore_spherical[0], Vpcore_spherical[1], Vpcore_spherical[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    VpcoreSphericalPara, VpcoreSphericalPerp = VpcoreSphericalBaligned[0], np.sqrt(VpcoreSphericalBaligned[1]**2 + VpcoreSphericalBaligned[2]**2)
    VpbeamSphericalBaligned = rotateVectorIntoFieldAligned(Vpbeam_spherical[0], Vpbeam_spherical[1], Vpbeam_spherical[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    VpbeamSphericalPara, VpbeamSphericalPerp = VpbeamSphericalBaligned[0], np.sqrt(VpbeamSphericalBaligned[1]**2 + VpbeamSphericalBaligned[2]**2)
    ValphaSphericalBaligned = rotateVectorIntoFieldAligned(Valpha_spherical[0], Valpha_spherical[1], Valpha_spherical[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    ValphaSphericalPara, ValphaSphericalPerp = ValphaSphericalBaligned[0], np.sqrt(ValphaSphericalBaligned[1]**2 + ValphaSphericalBaligned[2]**2)

    VpcoreTiedBaligned = rotateVectorIntoFieldAligned(Vpcore_tied[0], Vpcore_tied[1], Vpcore_tied[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    VpcoreTiedPara, VpcoreTiedPerp = VpcoreTiedBaligned[0], np.sqrt(VpcoreTiedBaligned[1]**2 + VpcoreTiedBaligned[2]**2)
    VpbeamTiedBaligned = rotateVectorIntoFieldAligned(Vpbeam_tied[0], Vpbeam_tied[1], Vpbeam_tied[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    VpbeamTiedPara, VpbeamTiedPerp = VpbeamTiedBaligned[0], np.sqrt(VpbeamTiedBaligned[1]**2 + VpbeamTiedBaligned[2]**2)
    ValphaTiedBaligned = rotateVectorIntoFieldAligned(Valpha_tied[0], Valpha_tied[1], Valpha_tied[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    ValphaTiedPara, ValphaTiedPerp = ValphaTiedBaligned[0], np.sqrt(ValphaTiedBaligned[1]**2 + ValphaTiedBaligned[2]**2)

    # Calculate the angle between drif speeds with respect to the magnetic field.
    VdriftDiag = np.array(Valpha_diag) - np.array(Vproton_diag)
    VdriftSpherical = np.array(Valpha_spherical) - np.array(Vproton_spherical)
    VdriftTied = np.array(Valpha_tied) - np.array(Vproton_tied)

    Theta_ProtonAlpha_Diag = angle_between_vectors(magF_SRF, VdriftDiag) * 180 / np.pi
    Theta_ProtonAlpha_Spherical = angle_between_vectors(magF_SRF, VdriftSpherical) * 180 / np.pi
    Theta_ProtonAlpha_Tied = angle_between_vectors(magF_SRF, VdriftTied) * 180 / np.pi

    # the ratio of max(Alpha VDF) / max(Proton VDF)
    # Sometimes when the separation is doing nonsense, this helps rule them out
    VDF_Amax_Pmax_tied = np.max(Alphas_tied.get_vdf()) / np.max(Protons_tied.get_vdf())
    VDF_Amax_Pmax_spherical = np.max(Alphas_spherical.get_vdf()) / np.max(Protons_spherical.get_vdf())
    VDF_Amax_Pmax_diag = np.max(Alphas_diag.get_vdf()) / np.max(Protons_diag.get_vdf())

    # Perpendicular speeds for total proton (not just core); needed for
    # the small-perpendicular-drift override in alignment_score.
    VprotonDiagBaligned = rotateVectorIntoFieldAligned(
        Vproton_diag[0], Vproton_diag[1], Vproton_diag[2],
        Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    VprotonDiagPerp = np.sqrt(VprotonDiagBaligned[1]**2 + VprotonDiagBaligned[2]**2)
    VprotonSphericalBaligned = rotateVectorIntoFieldAligned(
        Vproton_spherical[0], Vproton_spherical[1], Vproton_spherical[2],
        Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    VprotonSphericalPerp = np.sqrt(VprotonSphericalBaligned[1]**2 + VprotonSphericalBaligned[2]**2)
    VprotonTiedBaligned = rotateVectorIntoFieldAligned(
        Vproton_tied[0], Vproton_tied[1], Vproton_tied[2],
        Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    VprotonTiedPerp = np.sqrt(VprotonTiedBaligned[1]**2 + VprotonTiedBaligned[2]**2)

    # Perp drift magnitude for each type
    VdriftDiagPerp = np.abs(ValphaDiagPerp - VprotonDiagPerp)
    VdriftSphericalPerp = np.abs(ValphaSphericalPerp - VprotonSphericalPerp)
    VdriftTiedPerp = np.abs(ValphaTiedPerp - VprotonTiedPerp)

    Flag_Vperp_diag = VdriftDiagPerp / VprotonDiagPerp
    Flag_Vperp_Spherical = VdriftSphericalPerp / VprotonSphericalPerp
    Flag_Vperp_Tied = VdriftTiedPerp / VprotonTiedPerp

    # Put all angles and corresponding data in a list of tuples
    options = [
        (Flag_Vperp_diag, "Diag", Protons_diag, Alphas_diag, VDF_Amax_Pmax_diag),
        (Flag_Vperp_Spherical, "Spherical", Protons_spherical, Alphas_spherical, VDF_Amax_Pmax_spherical),
        (Flag_Vperp_Tied, "Tied", Protons_tied, Alphas_tied, VDF_Amax_Pmax_tied)
    ]

    # 1) Remove options with a too large alpha VDF peak
    # 2) If Alpha_VDF peak is larger than 25% Proton_VDF peak, not sensible! Remove!
    VDF_Amax_Pmax_threshold = 0.2
    valid_options = [
        opt for opt in options
        if (opt[4] is not None) and np.isfinite(opt[4]) and (opt[4] <= VDF_Amax_Pmax_threshold)
    ]

    # The one with the smallest flag is the best.
    if len(valid_options) == 0:
        best_option = min(options, key=lambda x: x[0])
    else:
        best_option = min(valid_options, key=lambda x: x[0])

    # Unpack the result
    # For this short interval, keep it at spherical
    # best_option = options[1]
    best_flag, which_one, Protons_current, Alphas_current, _ = best_option

    #print(f"{which_one} is better.")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plot_one(ax[0], ax[1], vel, vdf_corrected, Protons_current.get_vdf(component='core'), Protons_current.get_vdf(component='beam'), Alphas_current.get_vdf() / 4, 'Final')
    plt.savefig(result_path + '/Final_Result.png')
    plt.close()

    # Calculate Moments, and asve them.
    # Density.
    Npcore = cal_density_Spherical(Protons_current, component='core')
    Npbeam = cal_density_Spherical(Protons_current, component='beam')
    Nalpha = cal_density_Spherical(Alphas_current)

    # Velocity
    Vpcore = cal_bulk_velocity_Spherical(Protons_current, component='core')
    Vpbeam = cal_bulk_velocity_Spherical(Protons_current, component='beam')
    Valpha = cal_bulk_velocity_Spherical(Alphas_current)
    Vproton = cal_bulk_velocity_Spherical(Protons_current)
    Vap = Valpha - Vproton

    Vpcore_Baliged = rotateVectorIntoFieldAligned(Vpcore[0], Vpcore[1], Vpcore[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    VpcorePara, VpcorePerp = Vpcore_Baliged[0], np.sqrt(Vpcore_Baliged[1]**2 + Vpcore_Baliged[2]**2)
    Vpbeam_Baliged = rotateVectorIntoFieldAligned(Vpbeam[0], Vpbeam[1], Vpbeam[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    VpbeamPara, VpbeamPerp = Vpbeam_Baliged[0], np.sqrt(Vpbeam_Baliged[1]**2 + Vpbeam_Baliged[2]**2)
    Valpha_Baligned = rotateVectorIntoFieldAligned(Valpha[0], Valpha[1], Valpha[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    ValphaPara, ValphaPerp = Valpha_Baligned[0], np.sqrt(Valpha_Baligned[1]**2 + Valpha_Baligned[2]**2)
    Vproton_Baligned = rotateVectorIntoFieldAligned(Vproton[0], Vproton[1], Vproton[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    VprotonPara, VprotonPerp = Vproton_Baligned[0], np.sqrt(Vproton_Baligned[1]**2 + Vproton_Baligned[2]**2)

    # Temperature
    TparaProtonCore, TperpProtonCore = Temperature_para_perp(Protons_current, component='core')
    TparaProtonBeam, TperpProtonBeam = Temperature_para_perp(Protons_current, component='beam')
    TparaAlpha, TperpAlpha = Temperature_para_perp(Alphas_current)
    TparaProton, TperpProton = Temperature_para_perp(Protons_current)

    Tap_ratio = (TparaProton + 2 * TperpProton) / (TparaAlpha + 2 * TperpAlpha)

    # If the separation fails, in order to avoid the failure of the next separation, we use the previous separation as the initial value and return it to the next separation.
    if np.abs(best_flag) > 0.2:
        initial_means_next = initial_means
    else:
        initial_means_next = get_initial_means_from_objects(Protons_current, Alphas_current)

    Overlap_alpha_ref, Overlap_proton_ref = cal_overlap(Protons_current, Alphas_current)


    # Save the important parameter printed to a txt file.
    moments = {
        "Which one": which_one,
        "Bsrf0": magF_SRF[0],
        "Bsrf1": magF_SRF[1],
        "Bsrf2": magF_SRF[2],
        "Vsrf0_bulk": V_bulk_SRF[0],
        "Vsrf1_bulk": V_bulk_SRF[1],
        "Vsrf2_bulk": V_bulk_SRF[2],
        "Vsrf0_Proton": Vproton[0] / 1e3,
        "Vsrf1_Proton": Vproton[1] / 1e3,
        "Vsrf2_Proton": Vproton[2] / 1e3,
        "Vsrf0_Alpha": Valpha[0] / 1e3,
        "Vsrf1_Alpha": Valpha[1] / 1e3,
        "Vsrf2_Alpha": Valpha[2] / 1e3,
        "Npcore": Npcore / 1e6,
        "Npbeam": Npbeam / 1e6,
        "Nalpha": Nalpha / 1e6,
        "VpcorePara": VpcorePara / 1e3,
        "VpcorePerp": VpcorePerp / 1e3,
        "VpbeamPara": VpbeamPara / 1e3,
        "VpbeamPerp": VpbeamPerp / 1e3,
        "VprotonPara": VprotonPara / 1e3,
        "VprotonPerp": VprotonPerp / 1e3,
        "ValphaPara": ValphaPara / 1e3,
        "ValphaPerp": ValphaPerp / 1e3,
        "VA": VA,
        "TparaPcore": TparaProtonCore,
        "TperpPcore": TperpProtonCore,
        "TparaPbeam": TparaProtonBeam,
        "TperpPbeam": TperpProtonBeam,
        "TparaProton": TparaProton,
        "TperpProton": TperpProton,
        "TparaAlpha": TparaAlpha,
        "TperpAlpha": TperpAlpha,
        "Overlap_proton_ref": Overlap_proton_ref,
        "Overlap_alpha_ref": Overlap_alpha_ref,
    }

    # Save the moments, why not.
    with open(result_path+'/moments.txt', 'w') as f:
        for key, value in moments.items():
            f.write(f"{key}: {value}\n")

    # Save the results. Even fail, we save it.
    save_pickle(path=result_path+'/Protons.pkl', data=Protons_current)
    save_pickle(path=result_path+'/Alphas.pkl', data=Alphas_current)

    # Run one case with one-particle-noise removed.

    Protons_removed_1count = Protons_current.copy()
    vdf_pcore = Protons_removed_1count.get_vdf(component='core')
    vdf_pcore[vdf_pcore <= one_particle_noise_level] = 0

    vdf_pbeam = Protons_removed_1count.get_vdf(component='beam')
    vdf_pbeam[vdf_pbeam <= one_particle_noise_level] = 0

    Alphas_removed_1count = Alphas_current.copy()

    vdf_alpha = Alphas_removed_1count.get_vdf()
    vdf_alpha[vdf_alpha <= one_particle_noise_level] = 0

    Protons_removed_1count.set_vdf(vdf_pcore, component='core')
    Protons_removed_1count.set_vdf(vdf_pbeam, component='beam')
    Alphas_removed_1count.set_vdf(vdf_alpha)

    return initial_means_next

def parallelised_all_process(idx_time_list, initial_means, n_processes):
    """
    A parallelised version of all_process function.
    """

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
      futures = [executor.submit(all_process, idx, initial_means) for idx in idx_time_list]
      results = [f.result() for f in futures]
    avg = np.mean(np.stack(results, axis=0), axis=0)

    return avg  # return avg for the next initial means.


def calculate_noise_for_day(yymmdd):
    """
    v1: Calculate one-particle noise level for a single day.
    Called just-in-time before processing that day's data.
    """
    if os.path.exists(f'result/SO/{yymmdd}/one_particle_noise_level.npz'):
        print(f'One-particle noise level for {yymmdd} already exists, skip calculation.')
        return

    print(f'Calculating one-particle noise level for {yymmdd}...')
    data_list = os.listdir(f'data/SO/{yymmdd}')
    vdf_fname = next(file for file in data_list if 'pas-vdf' in file and not file.startswith('._'))
    count_fname = next(file for file in data_list if 'pas-3d' in file and not file.startswith('._'))
    vdf_cdffile = pycdf.CDF(f'data/SO/{yymmdd}/{vdf_fname}')
    count_cdffile = pycdf.CDF(f'data/SO/{yymmdd}/{count_fname}')
    if os.path.exists(f'result/SO/{yymmdd}') is False:
        os.makedirs(f'result/SO/{yymmdd}')
    one_particle_noise_level = OneParticleNoiseLevel(count_cdffile, vdf_cdffile)
    np.savez(f'result/SO/{yymmdd}/one_particle_noise_level.npz', noise_level=one_particle_noise_level)
    vdf_cdffile.close()
    count_cdffile.close()
    print(f'Done calculating noise for {yymmdd}.')


def main():
    total_tstart = time.time()

    # Specify the resolution of PAS during your interval
    # The time resolution of measurement, this is needed for checking if there is irregular data gap.
    dt_seconds = 1.0
    # The time resolution that you want to use, please make sure it is multiple of the actual dataresolution.
    dt_wanted = 4.0

    # t start should be the the time of the initial separation.
    t_start = datetime(2024, 1, 1, 5, 0, 1)
    hhmmss_start = t_start.strftime("%H%M%S")
    yymmdd_start = t_start.strftime('%Y%m%d')
    t_end = datetime(2024, 1, 31, 23, 59, 59)

    # block length in seconds
    block_length = 120  # in seconds
    block_size = int(block_length // dt_wanted)  # since data is at 4s resolution

    # Number of parallel processes, for maximum efficiency, we recommend it to be able to be evenly divided by the block size.
    n_processes = 30  # Use half of the block size for processes. If you have enough CPU cores, you can increase this number.

    # Initial Values
    Protons_initial = read_pickle(f'result/SO/{yymmdd_start}/Particles/Ions/{hhmmss_start}/Protons.pkl')
    Alphas_initial = read_pickle(f'result/SO/{yymmdd_start}/Particles/Ions/{hhmmss_start}/Alphas.pkl')
    initial_means = get_initial_means_from_objects(Protons_initial, Alphas_initial)

    t_start = t_start + timedelta(seconds=dt_seconds)  # Move to the next time slice for processing.
    yymmdd_lst = days_between(t_start, t_end)

    # ========================================================================
    # v1: DAY-BY-DAY PIPELINE
    # For each day: check data → calculate noise → run GMM blocks
    # This allows downloading Day N+1 while Day N is being processed.
    # ========================================================================
    for day_idx, yymmdd in enumerate(yymmdd_lst):
        day_tstart = time.time()
        print(f"\n{'='*60}")
        print(f"DAY {day_idx+1}/{len(yymmdd_lst)}: {yymmdd}")
        print(f"{'='*60}")

        # ----------------------------------------------------------------
        # Step 0: Wait for data if still downloading
        # ----------------------------------------------------------------
        if not wait_for_data(yymmdd, timeout=3600, poll_interval=30):
            print(f"[SKIP] Data never arrived for {yymmdd}, moving to next day.")
            continue

        # ----------------------------------------------------------------
        # Step 1: Check data for this day (collect & filter idx_times)
        # ----------------------------------------------------------------
        idx_times = collect_day_idx_times(yymmdd, t_start, t_end, dt_seconds)

        if not idx_times:
            print(f"[SKIP] No valid data for {yymmdd}, moving to next day.")
            continue

        # Keep the 'wanted' time resolution
        filtered_idx_times = resample_to_idx_time(idx_times, target_interval=dt_wanted)
        print(f"{len(filtered_idx_times)} time slices after resampling to {dt_wanted}s.")

        if not filtered_idx_times:
            print(f"[SKIP] No data after resampling for {yymmdd}, moving to next day.")
            continue

        # ----------------------------------------------------------------
        # Step 2: Calculate one-particle noise for this day (just-in-time)
        # ----------------------------------------------------------------
        calculate_noise_for_day(yymmdd)

        # ----------------------------------------------------------------
        # Step 2.5: Preload day-constant data (grids, epochs, filenames)
        # Workers inherit _day_data via fork, skipping redundant CDF reads.
        # ----------------------------------------------------------------
        preload_day_data(yymmdd)

        # ----------------------------------------------------------------
        # Step 3: Run GMM blocks for this day
        # ----------------------------------------------------------------
        n_blocks = int(np.ceil(len(filtered_idx_times) / block_size))
        for i in range(0, len(filtered_idx_times), block_size):
            block_num = i // block_size + 1
            idx_time_list = filtered_idx_times[i:i + block_size]
            if not idx_time_list:
                continue
            print(f"  Block {block_num}/{n_blocks} ({len(idx_time_list)} slices)...")
            initial_means = parallelised_all_process(
                idx_time_list, initial_means=initial_means, n_processes=n_processes)

        day_tend = time.time()
        print(f"Day {yymmdd} completed in {day_tend - day_tstart:.1f} seconds.")

    total_tend = time.time()
    print(f'\nTotal time used: {total_tend - total_tstart:.1f} seconds')

    return 0

if __name__ == "__main__":
    main()
