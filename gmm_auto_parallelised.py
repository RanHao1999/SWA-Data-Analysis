import os, sys
import concurrent.futures

import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from spacepy import pycdf
import astropy.units as u
import bisect
import scipy.ndimage

from gmm_resample import *
from Funcs import *
from SolarWindPack import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

np.set_printoptions(precision=2, suppress=True)
plt.rcParams.update({'font.size': 13, 'figure.dpi': 100})
print('All imports OK.')

# === CONFIG ===
YYMMDD = '20240508'
T_START_ISO = '2024-05-08 00:00:00'
T_END_ISO = '2024-05-08 23:59:59'
DT_WANTED = 4.0
N_PROCESSES = 20
_PLOT = False
# === END CONFIG ===


def wait_for_data(yymmdd, timeout=3600, poll_interval=30):
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
    if tstart > tend:
        raise ValueError("tstart should be smaller than tend.")
    result = [[idx, epoch[idx]] for idx in range(len(epoch)) if tstart <= epoch[idx] <= tend]
    return result


# Valid PAS operational cadences (s).  PAS can switch mode mid-day
# (e.g. 4 s -> 2 s), so slices at ANY of these cadences are kept —
# resample_to_idx_time() thins them to DT_WANTED downstream anyway.
ALLOWED_CADENCES = (4.0, 2.0, 1.0)


def find_irregular_times(tstart, tend, epoch_vdf, allowed=ALLOWED_CADENCES, tol=0.5):
    """Flag slices whose dt to the next slice matches no allowed cadence.

    dt larger than max(allowed)+tol is treated as a data gap and passes
    (same as the old one-sided check, which never flagged large dt).
    """
    if tstart > tend:
        raise ValueError("tstart should be smaller than tend.")
    bad_indices, bad_dts = [], []
    idx_in_interval = [i for i in range(len(epoch_vdf)) if tstart <= epoch_vdf[i] <= tend]
    gap_threshold = max(allowed) + tol
    for k in range(len(idx_in_interval) - 1):
        i, j = idx_in_interval[k], idx_in_interval[k+1]
        dt = (epoch_vdf[j] - epoch_vdf[i]).total_seconds()
        if dt >= gap_threshold:
            continue  # data gap — keep
        if not any(abs(dt - c) <= tol for c in allowed):
            bad_indices.append(i)
            bad_dts.append(dt)
    return bad_indices, bad_dts


def detect_cadence(epoch_vdf, t_start, t_end, n_samples=10):
    """Auto-detect PAS cadence from the first N consecutive time differences.

    Returns the median dt (seconds) rounded to 1 decimal place, or None
    if there are fewer than 2 slices in the interval.
    """
    idx_in_interval = [i for i in range(len(epoch_vdf))
                       if t_start <= epoch_vdf[i] <= t_end]
    if len(idx_in_interval) < 2:
        return None
    dts = []
    for k in range(min(n_samples, len(idx_in_interval) - 1)):
        i, j = idx_in_interval[k], idx_in_interval[k + 1]
        dt = (epoch_vdf[j] - epoch_vdf[i]).total_seconds()
        dts.append(dt)
    return round(float(np.median(dts)), 1)


def days_between(t_start, t_end):
    day = t_start.date()
    end = t_end.date()
    return [(day := day + timedelta(days=1)) and (day - timedelta(days=1)).strftime("%Y%m%d")
            for _ in range((end - day).days + 1)]


def collect_day_idx_times(yymmdd, t_start, t_end):
    print("\n=== Checking data for", yymmdd, "===")
    data_list = os.listdir(f"data/SO/{yymmdd}")
    vdf_fname = next(f for f in data_list if 'pas-vdf' in f and not f.startswith('._'))
    vdf_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{vdf_fname}")
    epoch_vdf = vdf_cdffile['Epoch'][...]

    mag_fname = next(f for f in data_list if 'mag-srf' in f and not f.startswith('._'))
    mag_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{mag_fname}")
    epoch_mag = mag_cdffile['EPOCH'][...]

    t_start_thisday = max(t_start, epoch_mag[0])
    t_end_thisday = min(t_end, epoch_mag[-1])

    idx_times = FindIndexinInterval(t_start_thisday, t_end_thisday, epoch_vdf)
    print(len(idx_times), "time indices found in total.")

    dt_seconds = detect_cadence(epoch_vdf, t_start_thisday, t_end_thisday)
    if dt_seconds is not None:
        print(f"Auto-detected PAS cadence at start of interval: {dt_seconds} s "
              f"(informational — any of {ALLOWED_CADENCES} s accepted)")
    bad_idx, deltas = find_irregular_times(t_start_thisday, t_end_thisday, epoch_vdf)
    print("Irregular intervals:", len(bad_idx))
    for idx, dt in zip(bad_idx[:20], deltas[:20]):
        print(f"Index {idx}: dt = {dt:.3f} s  ({epoch_vdf[idx]} -> {epoch_vdf[idx+1]})")
    if len(bad_idx) > 20:
        print(f"... and {len(bad_idx) - 20} more irregular slices (not listed).")
    bad_set = set(bad_idx)
    idx_times = [item for item in idx_times if item[0] not in bad_set]
    print(len(idx_times), "indices left after filtering.")

    vdf_cdffile.close()
    mag_cdffile.close()
    return idx_times


def resample_to_idx_time(idx_times, target_interval=4.0):
    if not idx_times:
        return []
    resampled_list = [idx_times[0]]
    last_kept_time = idx_times[0][1]
    for i in range(1, len(idx_times)):
        current_idx, current_time = idx_times[i]
        time_diff = (current_time - last_kept_time).total_seconds()
        if time_diff >= target_interval:
            resampled_list.append([current_idx, current_time])
            last_kept_time = current_time
    return resampled_list


def remove_noise(vdf, min_cluster_size=5, connectivity=1, positive_threshold=0.0):
    positive_mask = np.isfinite(vdf) & (vdf > positive_threshold)
    if not np.any(positive_mask):
        return vdf.copy()
    structure = scipy.ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
    labels, num_labels = scipy.ndimage.label(positive_mask, structure=structure)
    if num_labels == 0:
        return vdf.copy()
    label_sizes = np.bincount(labels.ravel(), minlength=num_labels + 1)
    keep_label = np.zeros(num_labels + 1, dtype=bool)
    keep_label[1:] = label_sizes[1:] >= min_cluster_size
    keep_mask = keep_label[labels]
    cleaned = vdf.copy()
    cleaned[~keep_mask] = 0
    return cleaned


def log10_vdf(vdf):
    vdf = np.array(vdf)
    mask = vdf > 0
    result = np.zeros_like(vdf)
    result[mask] = np.log10(vdf[mask])
    return result


def log10_1D_dist(vel, vdf):
    y = log10_vdf(np.sum(vdf, axis=(0, 1)))
    mask = y != 0
    return vel[mask], y[mask]


def plot_one(ax1, ax2, vel, vdf_total, f_core, f_beam, f_alpha, co_type):
    x, y = log10_1D_dist(vel, vdf_total)
    ax1.plot(x, y, label='Total', color='black')
    ax1.scatter(x, y, s=20, color='black', marker='s')
    for f_i, lbl, lc in [(f_core, 'Core', 'red'), (f_beam, 'Beam', 'blue'), (f_alpha, 'Alpha', 'green')]:
        xs, ys = log10_1D_dist(vel, f_i)
        ax1.plot(xs, ys, label=lbl, color=lc)
        ax1.scatter(xs, ys, s=10, color=lc)
    ax1.set_ylim(-12, -5)
    ax1.set_title(co_type)
    ax1.set_xlabel('Vel [km/s]')
    ax1.set_ylabel('log10(VDF)')
    ax1.legend()

    x, y = log10_1D_dist(vel, vdf_total)
    ax2.plot(x, y, label='Total', color='black')
    ax2.scatter(x, y, s=20, color='black', marker='s')
    xs, ys = log10_1D_dist(vel, f_core + f_beam)
    ax2.plot(xs, ys, label='Proton', color='red')
    ax2.scatter(xs, ys, s=10, color='red')
    xs, ys = log10_1D_dist(vel, f_alpha)
    ax2.plot(xs, ys, label='Alpha', color='green')
    ax2.scatter(xs, ys, s=10, color='green')
    ax2.set_ylim(-12, -5)
    ax2.set_xlabel('Vel [km/s]')
    ax2.set_ylabel('log10(VDF)')
    ax2.legend()
    return 0


# Module-level cache for day-constant data
_day_data = None


def preload_day_data(yymmdd):
    global _day_data
    print(f"  Preloading day-constant data for {yymmdd}...")
    data_list = os.listdir(f"data/SO/{yymmdd}")
    vdf_fname = next(f for f in data_list if "pas-vdf" in f and not f.startswith("._"))
    grnd_fname = next(f for f in data_list if "pas-grnd-mom" in f and not f.startswith("._"))
    mag_fname = next(f for f in data_list if "mag-srf-normal" in f and not f.startswith("._"))

    vdf_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{vdf_fname}")
    mag_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{mag_fname}")

    qp = 1.60217662e-19
    mp = 1.6726219e-27
    vel_raw = np.sqrt(2 * vdf_cdffile["Energy"][...] * qp / mp) / 1e3
    theta_raw = vdf_cdffile["Elevation"][...]
    phi_raw = vdf_cdffile["Azimuth"][...]
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



def auto_initial_means(vdf_corrected, vel, theta, phi, magF_SRF, VA):
    """
    Auto-detect alpha initial via thermal-safety bulk-velocity method.

    Same logic as GMM_Resample_Tutorial but automated:
    1. Find proton peak (argmax — protons dominate by density)
    2. Measure proton core width from low-speed HWHM of 1D VDF
    3. Define dividing boundary: Vp + 3×dv_hwhm + 1.5 * VA
       (thermal width clears core, Alfvén speed clears beam)
    4. Split VDF at boundary, compute BULK VELOCITY of the high-speed
       region → alpha initial for GMM

    Returns initial_means_3d, Vp_bulk (SRF), Va_bulk (SRF).
    """
    SAFETY_FACTOR = 3.0

    # ---- Proton peak: 3D global max ----
    i_phi_p, i_theta_p, i_energy_p = np.unravel_index(
        np.argmax(vdf_corrected), vdf_corrected.shape)

    # Proton peak SRF velocity (single pixel — protons are simple)
    V_SRF_x_p = (-vel[i_energy_p] * np.cos(theta[i_theta_p]) * np.cos(phi[i_phi_p])).value
    V_SRF_y_p = ( vel[i_energy_p] * np.cos(theta[i_theta_p]) * np.sin(phi[i_phi_p])).value
    V_SRF_z_p = (-vel[i_energy_p] * np.sin(theta[i_theta_p])).value

    # ---- FAC basis ----
    (Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz) = fieldAlignedCoordinates(
        magF_SRF[0], magF_SRF[1], magF_SRF[2])

    # ---- Measure dv_hwhm on LOW-SPEED side of 1D VDF ----
    f_1d = np.sum(vdf_corrected, axis=(0, 1))
    f_peak_val = f_1d[i_energy_p]
    i_half = None
    for j in range(i_energy_p, len(vel)):
        if f_1d[j] <= f_peak_val / 2.0:
            i_half = j
            break
        if f_1d[j] == 0 and j > i_energy_p + 2:
            i_half = j - 1
            break
    if i_half is None:
        dv_hwhm = 50.0
    else:
        dv_hwhm = abs(vel[i_energy_p].value - vel[i_half].value)
        dv_hwhm = max(dv_hwhm, 5.0)

    # ---- Dividing boundary ----
    v_alpha_min = vel[i_energy_p].value + SAFETY_FACTOR * 2 * dv_hwhm + 2.0 * VA    # From Harry's paper, 2.0 VA is an upper bound

    # ---- Split VDF, compute alpha velocity ----
    i_alpha = np.where(vel.value >= v_alpha_min)[0]

    if len(i_alpha) == 0 or not np.any(vdf_corrected[:, :, i_alpha] > 0):
        # Fallback: hottest pixel at the fastest measured bin.
        # When the dividing boundary exceeds all measured bins, take the
        # single brightest pixel at the highest-speed bin with any signal.
        has_signal = np.any(vdf_corrected > 0, axis=(0, 1))
        fastest_bin = np.min(np.where(has_signal)[0])
        vdf_slice = vdf_corrected[:, :, fastest_bin]
        i_az_hot, i_el_hot = np.unravel_index(np.argmax(vdf_slice), vdf_slice.shape)
        v_mag = vel[fastest_bin].value
        th_deg = theta[i_el_hot].value
        ph_deg = phi[i_az_hot].value
        th_rad = np.radians(th_deg)
        ph_rad = np.radians(ph_deg)
        vx = -v_mag * np.cos(th_rad) * np.cos(ph_rad)
        vy =  v_mag * np.cos(th_rad) * np.sin(ph_rad)
        vz = -v_mag * np.sin(th_rad)
        Va_bulk = np.array([vx, vy, vz])
    else:
        f_alpha_region = np.zeros_like(vdf_corrected)
        f_alpha_region[:, :, i_alpha] = vdf_corrected[:, :, i_alpha]
        Alphas_init = SolarWindParticle(
            'alpha', time=None, magfield=magF_SRF,
            grid=[theta.value, phi.value, vel.value * 1e3], coord_type='Spherical')
        Alphas_init.set_vdf(f_alpha_region)
        Va_bulk = cal_bulk_velocity_Spherical(Alphas_init) / 1e3  # km/s

    # Proton: peak pixel → FAC.  Alpha: bulk velocity of region → FAC.
    Vp_peak = np.array([V_SRF_x_p, V_SRF_y_p, V_SRF_z_p])
    Vp_BA = rotateVectorIntoFieldAligned(
        Vp_peak[0], Vp_peak[1], Vp_peak[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    Va_BA = rotateVectorIntoFieldAligned(
        Va_bulk[0], Va_bulk[1], Va_bulk[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)

    diff_alpha_core = np.array(Va_BA) - np.array(Vp_BA)

    initial_means = np.array([
        [0.0, 0.0, 0.0],        # core at rest in proton bulk frame
        [VA, 0.0, 0.0],        # beam placeholder — VA filled in by caller
        diff_alpha_core,          # alpha: bulk velocity of region above boundary
    ])

    return initial_means, Vp_peak, Va_bulk


def compute_flag_vperp(f_sorted, tslice_vdf, magF_SRF, theta, phi, vel,
                       Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz):
    """Compute Flag_Vperp and f_alpha/f_proton ratio for a GMM fit."""
    f_alpha, f_beam, f_core = f_sorted
    f_ratio = np.max(f_alpha) / np.max(f_core + f_beam)
    try:
        Protons = SolarWindParticle(
            'proton', time=tslice_vdf, magfield=magF_SRF,
            grid=[theta.value, phi.value, vel.value * 1e3], coord_type='Spherical')
        Protons.set_vdf(f_core, 'core')
        Protons.set_vdf(f_beam, 'beam')
        Alphas = SolarWindParticle(
            'alpha', time=tslice_vdf, magfield=magF_SRF,
            grid=[theta.value, phi.value, vel.value * 1e3 / np.sqrt(2)], coord_type='Spherical')
        Alphas.set_vdf(f_alpha * 4)
        Vp = cal_bulk_velocity_Spherical(Protons) / 1e3
        Va = cal_bulk_velocity_Spherical(Alphas) / 1e3
        Vp_BA = rotateVectorIntoFieldAligned(Vp[0], Vp[1], Vp[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
        Va_BA = rotateVectorIntoFieldAligned(Va[0], Va[1], Va[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
        Vp_para = Vp_BA[0]
        Vp_perp = np.sqrt(Vp_BA[1]**2 + Vp_BA[2]**2)
        Va_perp = np.sqrt(Va_BA[1]**2 + Va_BA[2]**2)
        flag = abs(Va_perp - Vp_perp) / abs(Vp_para) if abs(Vp_para) > 1.0 else np.inf
    except Exception:
        flag = np.inf
    return flag, f_ratio


def all_process(idx_time):
    """
    Process a single time slice.  Fully self-contained — auto-detects
    initial means from the VDF itself, no external state needed.
    """
    tsliceindex_vdf = idx_time[0]
    tslice_vdf = idx_time[1]

    yymmdd = tslice_vdf.strftime('%Y%m%d')
    hhmmss = tslice_vdf.strftime('%H%M%S')
    result_path = f'result/SO/{yymmdd}/Particles/Ions_auto_resample/{hhmmss}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    dd = _day_data

    vdf_cdffile = grnd_cdffile = mag_cdffile = None
    try:
        if dd is not None and dd["yymmdd"] == yymmdd:
            vdf_fname = dd["vdf_fname"]
            grnd_fname = dd["grnd_fname"]
            mag_fname = dd["mag_fname"]
            one_particle_noise_level = dd["noise_level"]
            epoch_vdf = dd["epoch_vdf"]
            epoch_mag = dd["epoch_mag"]
            qp = 1.60217662e-19
            mp = 1.6726219e-27
            vel = dd["vel_raw"] * (u.km / u.s)
            theta = dd["theta_raw"] * u.deg
            phi = dd["phi_raw"] * u.deg
            vdf_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{vdf_fname}")
            grnd_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{grnd_fname}")
            mag_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{mag_fname}")
        else:
            data_list = os.listdir(f"data/SO/{yymmdd}")
            vdf_fname = next(f for f in data_list if "pas-vdf" in f and not f.startswith("._"))
            grnd_fname = next(f for f in data_list if "pas-grnd-mom" in f and not f.startswith("._"))
            mag_fname = next(f for f in data_list if "mag-srf-normal" in f and not f.startswith("._"))
            vdf_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{vdf_fname}")
            grnd_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{grnd_fname}")
            mag_cdffile = pycdf.CDF(f"data/SO/{yymmdd}/{mag_fname}")
            loaded_data = np.load(f"result/SO/{yymmdd}/one_particle_noise_level.npz")
            one_particle_noise_level = loaded_data["noise_level"]
            epoch_vdf = vdf_cdffile["Epoch"][...]
            epoch_mag = mag_cdffile["EPOCH"][...]
            qp = 1.60217662e-19
            mp = 1.6726219e-27
            vel = np.sqrt(2 * vdf_cdffile["Energy"][...] * qp / mp) / 1e3 * (u.km / u.s)
            theta = vdf_cdffile["Elevation"][...] * u.deg
            phi = vdf_cdffile["Azimuth"][...] * u.deg

        # --- Magnetic field ---
        try:
            t0 = epoch_vdf[tsliceindex_vdf] - timedelta(seconds=0.5)
            t1 = epoch_vdf[tsliceindex_vdf] + timedelta(seconds=0.5)
            i0 = bisect.bisect_left(epoch_mag, t0)
            i1 = bisect.bisect_left(epoch_mag, t1)
            magF_SRF = mag_cdffile["B_SRF"][i0:i1].mean(axis=0)
        except Exception:
            idx_mag = bisect.bisect_left(epoch_mag, epoch_vdf[tsliceindex_vdf])
            magF_SRF = mag_cdffile["B_SRF"][idx_mag - 1]

        if not np.all(np.isfinite(magF_SRF)):
            return

        V_bulk_SRF = grnd_cdffile["V_SRF"][tsliceindex_vdf]
        vdf_raw = vdf_cdffile["vdf"][tsliceindex_vdf]

        # Alfven speed
        B_mag = np.sqrt(np.sum(magF_SRF**2)) * 1e-9
        density = grnd_cdffile["N"][tsliceindex_vdf] * 1e6
        mu0 = 4 * np.pi * 1e-7
        mp = 1.67e-27
        VA = B_mag / np.sqrt(mu0 * density * mp) / 1000.0
    finally:
        for _f in (vdf_cdffile, grnd_cdffile, mag_cdffile):
            if _f is not None:
                try:
                    _f.close()
                except Exception:
                    pass

    # --- Clean VDF ---
    vdf = vdf_raw.copy()
    vdf = remove_noise(vdf)

    # --- FAC coordinate transform ---
    vel_kms = vel.value
    v_bulk = np.asarray(V_bulk_SRF, dtype=float)

    t_rad = np.radians(theta.value)[np.newaxis, :, np.newaxis]
    p_rad = np.radians(phi.value)[:, np.newaxis, np.newaxis]
    v3d   = vel_kms[np.newaxis, np.newaxis, :]

    cos_t = np.cos(t_rad); sin_t = np.sin(t_rad)
    cos_p = np.cos(p_rad); sin_p = np.sin(p_rad)

    vx = -v3d * cos_t * cos_p - v_bulk[0]
    vy =  v3d * cos_t * sin_p - v_bulk[1]
    vz = -v3d * sin_t          - v_bulk[2]

    (Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz) = fieldAlignedCoordinates(
        magF_SRF[0], magF_SRF[1], magF_SRF[2])
    (V_para, V_perp1, V_perp2) = rotateVectorIntoFieldAligned(
        vx, vy, vz, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)

    # --- Auto-detect initial means ---
    init_means, Vp_SRF, Va_SRF = auto_initial_means(
        vdf, vel, theta, phi, magF_SRF, VA)

    # --- GMM fitting ---
    COV_TYPES = ['full', 'tied', 'diag', 'spherical']
    NonZeroCount = np.count_nonzero(vdf)
    N_SAMPLES = max(50000, NonZeroCount * 50)
    RANDOM_STATE = 42
    fac_kwargs = dict(Nx=Nx, Ny=Ny, Nz=Nz, Px=Px, Py=Py, Pz=Pz, Qx=Qx, Qy=Qy, Qz=Qz)

    all_results = {}
    for co_type in COV_TYPES:
        try:
            f_sorted, gmm_info, probas, _ = cal_GMM_resampled(
                V_para, V_perp1, V_perp2, vdf,
                vel, theta, phi,
                init_means,
                co_type=co_type,
                n_samples=N_SAMPLES,
                random_state=RANDOM_STATE,
                intra_bin='interp',
                V_bulk_SRF=V_bulk_SRF,
                bin_avg_posterior=False,
                **fac_kwargs,
            )
            all_results[co_type] = (f_sorted, gmm_info, probas)
        except Exception as e:
            all_results[co_type] = None

    # --- Select best covariance type ---
    candidates = []
    for co_type in COV_TYPES:
        if all_results.get(co_type) is not None:
            flag, fr = compute_flag_vperp(
                all_results[co_type][0], tslice_vdf, magF_SRF, theta, phi, vel,
                Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
            candidates.append((co_type, flag, fr))

    valid = [(ct, fl, fr) for ct, fl, fr in candidates if fr <= 0.2 and np.isfinite(fl)]
    if not valid:
        valid = [(ct, fl, fr) for ct, fl, fr in candidates if np.isfinite(fl)]

    if valid:
        best_type, best_flag, best_fr = min(valid, key=lambda x: x[1])
        best_f_sorted, best_gmm_info, best_probas = all_results[best_type]
    else:
        return  # no valid fit

    fa, fb, fc = best_f_sorted

    # --- Build output objects ---
    Protons_out = SolarWindParticle(
        'proton', time=tslice_vdf, magfield=magF_SRF,
        grid=[theta.value, phi.value, vel.value * 1e3], coord_type='Spherical')
    Protons_out.set_vdf(fc, 'core')
    Protons_out.set_vdf(fb, 'beam')

    Alphas_out = SolarWindParticle(
        'alpha', time=tslice_vdf, magfield=magF_SRF,
        grid=[theta.value, phi.value, vel.value * 1e3 / np.sqrt(2)], coord_type='Spherical')
    Alphas_out.set_vdf(fa * 4)

    # --- Save ---
    save_pickle(f'{result_path}/Protons.pkl', Protons_out)
    save_pickle(f'{result_path}/Alphas.pkl', Alphas_out)

    # --- Plot the best one ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    fig.suptitle(f'Best GMM Separation: {best_type} covariance, interp mode\n'
                f'Time: {tslice_vdf.strftime("%Y-%m-%d %H:%M:%S")}, ',
                fontsize=13, fontweight='bold')
    plot_one(axes[0], axes[1], vel, vdf, fc, fb, fa, best_type)
    plt.tight_layout()
    plt.savefig(f'{result_path}/Final_result.png', dpi=150, bbox_inches='tight')
    plt.close()


    # --- Diagnostic plot (optional) ---
    if _PLOT:
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        fig.suptitle(f'GMM: All 4 Covariance Types (interp mode)\n'
                     f'Time: {tslice_vdf.strftime("%Y-%m-%d %H:%M:%S")}',
                     fontsize=14, fontweight='bold')
        for col, co_type in enumerate(COV_TYPES):
            if all_results.get(co_type) is not None:
                fai, fbi, fci = all_results[co_type][0]
                plot_one(axes[0, col], axes[1, col], vel, vdf, fci, fbi, fai, co_type)
            else:
                axes[0, col].text(0.5, 0.5, 'FAILED', transform=axes[0, col].transAxes,
                                  ha='center', va='center', fontsize=14, color='gray')
            if col == 0:
                axes[0, col].set_ylabel('log10(VDF)')
                axes[1, col].set_ylabel('log10(VDF)')
        plt.tight_layout()
        plt.savefig(result_path + '/all_cov_types.png', dpi=150, bbox_inches='tight')
        plt.close()

    return 0


def parallelised_all_process(idx_time_list, n_processes):
    """Pure embarrassingly-parallel map over time slices."""
    total = len(idx_time_list)
    print(f"  Processing {total} slices with {n_processes} workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = {executor.submit(all_process, idx): i for i, idx in enumerate(idx_time_list)}
        done = 0
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"  [ERROR] Slice failed: {e}")
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                print(f"  [{done}/{total}] slices completed")


def calculate_noise_for_day(yymmdd):
    if os.path.exists(f'result/SO/{yymmdd}/one_particle_noise_level.npz'):
        print(f'One-particle noise level for {yymmdd} already exists, skip calculation.')
        return
    print(f'Calculating one-particle noise level for {yymmdd}...')
    data_list = os.listdir(f'data/SO/{yymmdd}')
    vdf_fname = next(f for f in data_list if 'pas-vdf' in f and not f.startswith('._'))
    count_fname = next(f for f in data_list if 'pas-3d' in f and not f.startswith('._'))
    vdf_cdffile = pycdf.CDF(f'data/SO/{yymmdd}/{vdf_fname}')
    count_cdffile = pycdf.CDF(f'data/SO/{yymmdd}/{count_fname}')
    if not os.path.exists(f'result/SO/{yymmdd}'):
        os.makedirs(f'result/SO/{yymmdd}')
    one_particle_noise_level = OneParticleNoiseLevel(count_cdffile, vdf_cdffile)
    np.savez(f'result/SO/{yymmdd}/one_particle_noise_level.npz', noise_level=one_particle_noise_level)
    vdf_cdffile.close()
    count_cdffile.close()
    print(f'Done calculating noise for {yymmdd}.')


def main():
    """Run GMM processing for a single day.  Config is read from module-level
    variables set in the # === CONFIG === block above."""
    total_tstart = time.time()

    t_start = datetime.fromisoformat(T_START_ISO)
    t_end   = datetime.fromisoformat(T_END_ISO)

    print(f"\n{'='*60}")
    print(f"DAY {YYMMDD}")
    print(f"{'='*60}")

    if not wait_for_data(YYMMDD, timeout=3600, poll_interval=30):
        print(f"[FATAL] Data never arrived for {YYMMDD}.")
        return 1

    idx_times = collect_day_idx_times(YYMMDD, t_start, t_end)
    if not idx_times:
        print(f"[SKIP] No valid data for {YYMMDD}.")
        return 0

    filtered = resample_to_idx_time(idx_times, target_interval=DT_WANTED)
    print(f"{len(filtered)} time slices after resampling to {DT_WANTED}s.")
    if not filtered:
        print(f"[SKIP] No data after resampling for {YYMMDD}.")
        return 0

    calculate_noise_for_day(YYMMDD)
    preload_day_data(YYMMDD)
    # Clamp workers to actually-available cores (leave headroom for other users).
    _avail = len(os.sched_getaffinity(0))
    _n_procs = min(N_PROCESSES, max(1, _avail - 2))
    if _n_procs != N_PROCESSES:
        print(f"  [ADAPT] Only {_avail} cores available → "
              f"using {_n_procs} workers (configured max: {N_PROCESSES})")
    parallelised_all_process(filtered, n_processes=_n_procs)

    total_tend = time.time()
    print(f"Day {YYMMDD} completed in {total_tend - total_tstart:.1f} seconds.")
    return 0


if __name__ == "__main__":
    main()
