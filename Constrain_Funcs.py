from scipy import ndimage
import pandas as pd
import numpy as np
import os
import h5py
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from Funcs import *
from SolarWindPack import *


def cal_overlap_features(protons, alphas, alpha_scale=1/4):
    """
    Calculate 1D speed-projected overlap features between proton and alpha VDFs.

    Returns
    -------
    features : dict
        overlap_alpha_density:
            Fraction of alpha intensity lying under the proton profile.
        overlap_proton_density:
            Fraction of proton intensity lying under the alpha profile.
        overlap_shape:
            Overlap between normalised proton and alpha speed profiles.
        alpha_to_proton_total:
            Total alpha/proton intensity ratio in the projected profile.
    """

    vdf_p = protons.get_vdf()
    vdf_a = alphas.get_vdf() * alpha_scale  # Apply scaling to alpha VDF

    # Collapse angular dimensions.
    p1d = np.nansum(vdf_p, axis=(0, 1))
    a1d = np.nansum(vdf_a, axis=(0, 1))

    # Remove negative or non-finite values, just in case.
    p1d = np.where(np.isfinite(p1d) & (p1d > 0), p1d, 0.0)
    a1d = np.where(np.isfinite(a1d) & (a1d > 0), a1d, 0.0)

    sum_p = np.sum(p1d)
    sum_a = np.sum(a1d)

    if sum_p <= 0 or sum_a <= 0:
        return {
            "overlap_alpha_density": np.nan,
            "overlap_proton_density": np.nan,
            "overlap_shape": np.nan,
            "alpha_to_proton_total": np.nan,
        }

    min_vdf = np.minimum(p1d, a1d)

    overlap_alpha_density = np.sum(min_vdf) / sum_a
    overlap_proton_density = np.sum(min_vdf) / sum_p

    p_norm = p1d / sum_p
    a_norm = a1d / sum_a

    overlap_shape = np.sum(np.minimum(p_norm, a_norm))

    return {
        "overlap_alpha_density": overlap_alpha_density,
        "overlap_proton_density": overlap_proton_density,
        "overlap_shape": overlap_shape,
        "alpha_to_proton_total": sum_a / sum_p,
    }


def _connectedness_single_species(
    f_self,
    f_other,
    species_name,
    frac_threshold=0.2,
    dynamic_range_decades=3.0,
    connectivity=1,
):
    """
    Compute connectedness diagnostics for one species in 3D VDF space.

    A voxel is considered valid support for the species if:
    1) it is above a dynamic floor relative to that species peak, and
    2) species fraction f_self/(f_self+f_other) exceeds frac_threshold.
    """

    f_self = np.asarray(f_self, dtype=float)
    f_other = np.asarray(f_other, dtype=float)

    # Clean bad values.
    f_self = np.where(np.isfinite(f_self) & (f_self > 0), f_self, 0.0)
    f_other = np.where(np.isfinite(f_other) & (f_other > 0), f_other, 0.0)

    prefix = f"{species_name}_"

    if np.nanmax(f_self) <= 0:
        return {
            f"{prefix}connectedness": np.nan,
            f"{prefix}n_components": 0,
            f"{prefix}largest_component_weight": 0.0,
            f"{prefix}total_mask_weight": 0.0,
            f"{prefix}fragment_weight_fraction": np.nan,
            f"{prefix}largest_component_size": 0,
            f"{prefix}total_mask_size": 0,
        }

    total = f_self + f_other
    frac_self = np.divide(
        f_self,
        total,
        out=np.zeros_like(f_self),
        where=total > 0,
    )

    self_floor = np.nanmax(f_self) * 10 ** (-dynamic_range_decades)
    support_mask = (f_self > self_floor) & (frac_self > frac_threshold)

    if np.sum(support_mask) == 0:
        return {
            f"{prefix}connectedness": 0.0,
            f"{prefix}n_components": 0,
            f"{prefix}largest_component_weight": 0.0,
            f"{prefix}total_mask_weight": 0.0,
            f"{prefix}fragment_weight_fraction": 1.0,
            f"{prefix}largest_component_size": 0,
            f"{prefix}total_mask_size": 0,
        }

    structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
    labelled, n_components = ndimage.label(support_mask, structure=structure)

    component_ids = np.arange(1, n_components + 1)
    component_weights = ndimage.sum(f_self, labelled, index=component_ids)
    component_sizes = ndimage.sum(support_mask.astype(int), labelled, index=component_ids)

    total_weight = np.sum(component_weights)

    if total_weight <= 0:
        connectedness = 0.0
        largest_weight = 0.0
        largest_size = 0
    else:
        largest_idx = np.argmax(component_weights)
        largest_weight = component_weights[largest_idx]
        largest_size = component_sizes[largest_idx]
        connectedness = largest_weight / total_weight

    return {
        f"{prefix}connectedness": float(connectedness),
        f"{prefix}n_components": int(n_components),
        f"{prefix}largest_component_weight": float(largest_weight),
        f"{prefix}total_mask_weight": float(total_weight),
        f"{prefix}fragment_weight_fraction": float(1.0 - connectedness),
        f"{prefix}largest_component_size": int(largest_size),
        f"{prefix}total_mask_size": int(np.sum(support_mask)),
    }


def check_connectedness(
    protons,
    alphas,
    alpha_scale=1/4,
    alpha_frac_threshold=0.2,
    proton_frac_threshold=0.5,
    dynamic_range_decades=3.0,
    connectivity=1,
):
    """
    Quantify connectedness for both alpha and proton VDFs.

    Returns one merged dictionary containing alpha_* and proton_* metrics,
    plus summary metrics across both species.
    """

    fp = np.asarray(protons.get_vdf(), dtype=float)
    fa = np.asarray(alphas.get_vdf(), dtype=float) * alpha_scale

    alpha_features = _connectedness_single_species(
        f_self=fa,
        f_other=fp,
        species_name="alpha",
        frac_threshold=alpha_frac_threshold,
        dynamic_range_decades=dynamic_range_decades,
        connectivity=connectivity,
    )

    proton_features = _connectedness_single_species(
        f_self=fp,
        f_other=fa,
        species_name="proton",
        frac_threshold=proton_frac_threshold,
        dynamic_range_decades=dynamic_range_decades,
        connectivity=connectivity,
    )

    alpha_conn = alpha_features["alpha_connectedness"]
    proton_conn = proton_features["proton_connectedness"]
    conn_values = np.array([alpha_conn, proton_conn], dtype=float)

    summary = {
        "min_species_connectedness": float(np.nanmin(conn_values)) if np.any(np.isfinite(conn_values)) else np.nan,
        "mean_species_connectedness": float(np.nanmean(conn_values)) if np.any(np.isfinite(conn_values)) else np.nan,
    }

    return {**alpha_features, **proton_features, **summary}



def _smoothness_single_species(
    f_1d,
    species_name,
    rel_floor=1e-3,
    smooth_sigma=1.0,
    roughness_tol_dex=0.20,
    peak_prominence_dex=0.25,
):
    """
    Compute smoothness diagnostics for one species' 1D velocity profile.

    Parameters
    ----------
    f_1d : array
        1D velocity profile (already collapsed from 3D VDF).
    species_name : str
        'alpha' or 'proton' for output key naming.
    rel_floor : float
        Analyse region where f_1d > rel_floor * max(f_1d).
    smooth_sigma : float
        Gaussian smoothing width in velocity bins.
    roughness_tol_dex : float
        RMS residual tolerance in log10 space for scoring.
    peak_prominence_dex : float
        Prominence threshold in log10 space for peak detection.

    Returns
    -------
    dict
        Dictionary with {species}_* keys for jaggedness, smoothness, curvature,
        peak count, secondary peak ratio, and support size.
    """

    f_1d = np.asarray(f_1d, dtype=float)
    f_1d = np.where(np.isfinite(f_1d) & (f_1d > 0), f_1d, 0.0)

    prefix = f"{species_name}_"

    if np.nanmax(f_1d) <= 0:
        return {
            f"{prefix}jaggedness_rms_dex": np.nan,
            f"{prefix}smoothness_score": np.nan,
            f"{prefix}curvature_roughness": np.nan,
            f"{prefix}n_significant_peaks": 0,
            f"{prefix}secondary_peak_ratio": np.nan,
            f"{prefix}profile_n_points": 0,
        }

    f_max = np.nanmax(f_1d)
    floor = rel_floor * f_max

    # Support region where signal exceeds floor
    support = f_1d > floor

    if np.sum(support) < 5:
        return {
            f"{prefix}jaggedness_rms_dex": np.nan,
            f"{prefix}smoothness_score": 0.0,
            f"{prefix}curvature_roughness": np.nan,
            f"{prefix}n_significant_peaks": 0,
            f"{prefix}secondary_peak_ratio": np.nan,
            f"{prefix}profile_n_points": int(np.sum(support)),
        }

    # Use continuous range from first to last supported bin
    idx = np.where(support)[0]
    i0, i1 = idx[0], idx[-1] + 1
    f_seg = f_1d[i0:i1]

    # Log-space profile (clamp to floor to avoid log(0))
    logf = np.log10(np.maximum(f_seg, floor))

    # Smooth in log-space
    logf_smooth = gaussian_filter1d(logf, sigma=smooth_sigma, mode="nearest")

    # 1. RMS jaggedness: deviation from smoothed profile
    residual = logf - logf_smooth
    jaggedness_rms = np.sqrt(np.nanmean(residual**2))

    # Smoothness score: 1 = smooth, 0 = rough
    smoothness_score = 1.0 - jaggedness_rms / roughness_tol_dex
    smoothness_score = float(np.clip(smoothness_score, 0.0, 1.0))

    # 2. Curvature roughness: mean absolute second derivative (catches oscillations)
    if len(logf) >= 3:
        d2 = logf[2:] - 2 * logf[1:-1] + logf[:-2]
        curvature_roughness = float(np.nanmean(np.abs(d2)))
    else:
        curvature_roughness = np.nan

    # 3. Count significant peaks in smoothed log-profile
    peaks, _ = find_peaks(logf_smooth, prominence=peak_prominence_dex)
    n_peaks = len(peaks)

    # Secondary peak ratio (in linear space)
    if n_peaks >= 2:
        peak_values = f_seg[peaks]
        peak_values_sorted = np.sort(peak_values)[::-1]
        secondary_peak_ratio = float(peak_values_sorted[1] / peak_values_sorted[0])
    else:
        secondary_peak_ratio = 0.0

    return {
        f"{prefix}jaggedness_rms_dex": float(jaggedness_rms),
        f"{prefix}smoothness_score": smoothness_score,
        f"{prefix}curvature_roughness": curvature_roughness,
        f"{prefix}n_significant_peaks": int(n_peaks),
        f"{prefix}secondary_peak_ratio": float(secondary_peak_ratio),
        f"{prefix}profile_n_points": int(np.sum(support)),
    }


def check_smoothness(
    protons,
    alphas,
    rel_floor=1e-3,
    smooth_sigma=1.0,
    roughness_tol_dex=0.20,
    peak_prominence_dex=0.25,
):
    """
    Quantify smoothness for both alpha and proton VDFs.

    Returns one merged dictionary containing alpha_* and proton_* metrics,
    plus summary smoothness metrics.
    """

    fp = np.asarray(protons.get_vdf(), dtype=float)
    fa = np.asarray(alphas.get_vdf(), dtype=float)

    # Collapse angular dimensions
    fp_1d = np.nansum(fp, axis=(0, 1))
    fa_1d = np.nansum(fa, axis=(0, 1))

    alpha_features = _smoothness_single_species(
        f_1d=fa_1d,
        species_name="alpha",
        rel_floor=rel_floor,
        smooth_sigma=smooth_sigma,
        roughness_tol_dex=roughness_tol_dex,
        peak_prominence_dex=peak_prominence_dex,
    )

    proton_features = _smoothness_single_species(
        f_1d=fp_1d,
        species_name="proton",
        rel_floor=rel_floor,
        smooth_sigma=smooth_sigma,
        roughness_tol_dex=roughness_tol_dex,
        peak_prominence_dex=peak_prominence_dex,
    )

    alpha_smooth = alpha_features["alpha_smoothness_score"]
    proton_smooth = proton_features["proton_smoothness_score"]
    smooth_values = np.array([alpha_smooth, proton_smooth], dtype=float)

    summary = {
        "min_species_smoothness_score": float(np.nanmin(smooth_values)) if np.any(np.isfinite(smooth_values)) else np.nan,
        "mean_species_smoothness_score": float(np.nanmean(smooth_values)) if np.any(np.isfinite(smooth_values)) else np.nan,
    }

    return {**alpha_features, **proton_features, **summary}

def cal_alfven_speed_kms(B_nT, n_p_cm3, n_a_cm3=0.0):
    """
    Calculate Alfven speed in km/s.

    Parameters
    ----------
    B_nT : array-like, shape (3,) or float
        Magnetic field in nT.
    n_p_cm3 : float
        Proton density in cm^-3.
    n_a_cm3 : float
        Alpha density in cm^-3.

    Returns
    -------
    V_A : float
        Alfven speed in km/s.
    """
    mu0 = 4 * np.pi * 1e-7
    m_p = 1.6726219e-27
    m_a = 6.6464731e-27

    B_T = np.linalg.norm(B_nT) * 1e-9

    # cm^-3 to m^-3
    rho = (n_p_cm3 * 1e6) * m_p + (n_a_cm3 * 1e6) * m_a

    if rho <= 0 or not np.isfinite(rho):
        return np.nan

    return B_T / np.sqrt(mu0 * rho) * 1e3  


def soft_range_core(x, good_min, good_max, bad_min, bad_max, strictness=2.0):
    """
    Score a scalar from 0 to 1 using a stricter soft threshold.

    1 inside [good_min, good_max].
    Falls to 0 outside [bad_min, bad_max].
    Between the good and bad range, the score is squared to penalize weak cases more strongly.
    """

    if not np.isfinite(x):
        return 0.0

    if good_min <= x <= good_max:
        return 1.0

    if x < good_min:
        if x <= bad_min:
            return 0.0
        base = (x - bad_min) / (good_min - bad_min)
        return float(np.clip(base, 0.0, 1.0) ** strictness)

    if x > good_max:
        if x >= bad_max:
            return 0.0
        base = (bad_max - x) / (bad_max - good_max)
        return float(np.clip(base, 0.0, 1.0) ** strictness)

    return 0.0


# Backward-compatible alias.
soft_range_score = soft_range_core


def cal_moment_plausibility_features(
    protons,
    alphas,
    velocity_sign_convention=[-1, 1, -1],
    density_unit="cm-3",
    hard_reject_threshold=0.80,
):
    """
    Calculate moment-based plausibility features for proton-alpha separation.

    Uses existing moment functions:
        cal_density_Spherical
        cal_bulk_velocity_Spherical
        Temperature_para_perp

    Returns
    -------
    dict
        Moment features, a stricter plausibility score, and an approve/reject flag.
    """

    # Density
    n_p = cal_density_Spherical(protons)
    n_a = cal_density_Spherical(alphas)

    # Bulk velocity, assumed km/s if your grid velocity is km/s
    V_p = cal_bulk_velocity_Spherical(
        protons,
        velocity_sign_convention=velocity_sign_convention,
    )
    V_a = cal_bulk_velocity_Spherical(
        alphas,
        velocity_sign_convention=velocity_sign_convention,
    )

    # Temperatures in eV, assuming your temperature function is unit-correct
    Tp_para, Tp_perp = Temperature_para_perp(
        protons,
        velocity_sign_convention=velocity_sign_convention,
    )

    Ta_para, Ta_perp = Temperature_para_perp(
        alphas,
        velocity_sign_convention=velocity_sign_convention,
    )

    # Basic derived quantities
    n_ratio = n_a / n_p if n_p > 0 else np.nan

    dV_vec = V_a - V_p
    dV = np.linalg.norm(dV_vec)

    Vp_mag = np.linalg.norm(V_p)
    Va_mag = np.linalg.norm(V_a)

    drift_over_Vp = dV / Vp_mag if Vp_mag > 0 else np.nan

    # Alfven speed
    # This assumes density is in cm^-3 and B is in nT.
    # If your density unit is different, adapt this part.
    if density_unit == "cm-3":
        V_A = cal_alfven_speed_kms(protons.magfield, n_p, n_a)
    else:
        V_A = np.nan

    drift_over_VA = dV / V_A if np.isfinite(V_A) and V_A > 0 else np.nan

    # Temperature ratios
    TaTp_para = Ta_para / Tp_para if Tp_para > 0 else np.nan
    TaTp_perp = Ta_perp / Tp_perp if Tp_perp > 0 else np.nan

    alpha_anisotropy = Ta_perp / Ta_para if Ta_para > 0 else np.nan
    proton_anisotropy = Tp_perp / Tp_para if Tp_para > 0 else np.nan

    # Basic validity checks
    moment_values = np.array([
        n_p, n_a,
        *V_p, *V_a,
        Tp_para, Tp_perp,
        Ta_para, Ta_perp,
    ], dtype=float)

    all_finite = np.all(np.isfinite(moment_values))
    all_positive_core = (
        n_p > 0 and n_a > 0 and
        Tp_para > 0 and Tp_perp > 0 and
        Ta_para > 0 and Ta_perp > 0
    )

    # Soft scores.
    # These ranges are deliberately broad.
    # We are catching absurd failures, not doing physics selection.
    score_density = soft_range_core(
        n_ratio,
        good_min=0.005,
        good_max=0.08,
        bad_min=0.0,
        bad_max=0.25,
    )

    score_drift = soft_range_core(
        drift_over_VA,
        good_min=0.0,
        good_max=1.5,
        bad_min=0.0,
        bad_max=4.0,
    )

    score_Tpara_ratio = soft_range_core(
        TaTp_para,
        good_min=1.0,
        good_max=12.0,
        bad_min=0.1,
        bad_max=50.0,
    )

    score_Tperp_ratio = soft_range_core(
        TaTp_perp,
        good_min=1.0,
        good_max=12.0,
        bad_min=0.1,
        bad_max=30.0,
    )

    score_alpha_anis = soft_range_core(
        alpha_anisotropy,
        good_min=0.2,
        good_max=5.0,
        bad_min=0.03,
        bad_max=30.0,
    )

    score_proton_anis = soft_range_core(
        proton_anisotropy,
        good_min=0.2,
        good_max=5.0,
        bad_min=0.03,
        bad_max=30.0,
    )

    score_items = {
        "score_density": score_density,
        "score_drift": score_drift,
        "score_Tpara_ratio": score_Tpara_ratio,
        "score_Tperp_ratio": score_Tperp_ratio,
        "score_alpha_anisotropy": score_alpha_anis,
        "score_proton_anisotropy": score_proton_anis,
    }

    score_values = np.array(list(score_items.values()), dtype=float)
    bad_checks = [name for name, value in score_items.items() if (not np.isfinite(value)) or (value < hard_reject_threshold)]

    if not all_finite or not all_positive_core:
        moment_plausibility_score = 0.0
        moment_plausibility_mean_score = 0.0
        moment_plausibility_flag = "reject"
    else:
        moment_plausibility_score = float(np.nanmin(score_values))
        moment_plausibility_mean_score = float(np.nanmean(score_values))
        moment_plausibility_flag = "approve" if len(bad_checks) == 0 else "reject"

    return {
        # Densities
        "n_p": float(n_p),
        "n_a": float(n_a),
        "n_alpha_over_np": float(n_ratio),

        # Velocities
        "Vp_x": float(V_p[0]),
        "Vp_y": float(V_p[1]),
        "Vp_z": float(V_p[2]),
        "Va_x": float(V_a[0]),
        "Va_y": float(V_a[1]),
        "Va_z": float(V_a[2]),
        "Vp_mag": float(Vp_mag),
        "Va_mag": float(Va_mag),
        "alpha_proton_drift": float(dV),
        "alpha_proton_drift_over_Vp": float(drift_over_Vp),
        "V_A": float(V_A) if np.isfinite(V_A) else np.nan,
        "alpha_proton_drift_over_VA": float(drift_over_VA) if np.isfinite(drift_over_VA) else np.nan,

        # Temperatures
        "Tp_para": float(Tp_para),
        "Tp_perp": float(Tp_perp),
        "Ta_para": float(Ta_para),
        "Ta_perp": float(Ta_perp),
        "Ta_over_Tp_para": float(TaTp_para),
        "Ta_over_Tp_perp": float(TaTp_perp),
        "alpha_Tperp_over_Tpara": float(alpha_anisotropy),
        "proton_Tperp_over_Tpara": float(proton_anisotropy),

        # Individual scores
        **{k: float(v) for k, v in score_items.items()},

        # Final score and decision
        "moment_all_finite": bool(all_finite),
        "moment_all_positive": bool(all_positive_core),
        "moment_plausibility_score": float(moment_plausibility_score),
        "moment_plausibility_mean_score": float(moment_plausibility_mean_score),
        "moment_plausibility_flag": moment_plausibility_flag,
        "moment_plausibility_hard_reject_threshold": float(hard_reject_threshold),
        "moment_plausibility_bad_checks": bad_checks,
    }
