"""
Resampling-based GMM for VDF decomposition.

Instead of using [V_para, V_perp1, V_perp2, |V|, f] as a 5D feature vector
(which mixes velocity and phase space density — dimensionally inconsistent),
we:

1. Resample "particles" from the VDF: grid cells with high f produce many
   particles, grid cells with low f produce few.
2. Run GMM in PURE 3D velocity space (V_para, V_perp1, V_perp2).
3. Apply the fitted GMM posteriors back to the original VDF grid to
   partition it into core, beam, and alpha components.

This is physically sound: f is treated as a density (determining how many
particles to sample), not as a coordinate alongside velocity.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.interpolate import RegularGridInterpolator


def _build_fine_grid(vdf, ph_bin, th_bin, v_bin, upsample):
    """
    Build an upsampled grid and trilinearly interpolate the VDF onto it.

    Parameters
    ----------
    vdf : ndarray, shape (N_az, N_el, N_en)
        Original VDF on the coarse PAS grid.
    ph_bin, th_bin, v_bin : ndarray
        Coarse grid centres (azimuth [deg], elevation [deg], speed [km/s]).
    upsample : int
        Upsample factor per axis (e.g. 2 → 8× as many cells).

    Returns
    -------
    ph_fine, th_fine, v_fine : ndarray
        Fine grid centres.
    f_fine : ndarray, shape (N_az*upsample, N_el*upsample, N_en*upsample)
        Interpolated VDF on the fine grid.
    """
    n_ph, n_th, n_v = len(ph_bin), len(th_bin), len(v_bin)

    # Fine grid centres: linearly spaced within the same range
    ph_fine = np.linspace(ph_bin[0], ph_bin[-1], n_ph * upsample)
    th_fine = np.linspace(th_bin[0], th_bin[-1], n_th * upsample)
    v_fine  = np.linspace(v_bin[0],  v_bin[-1],  n_v  * upsample)

    # Trilinear interpolation
    vdf_clean = np.where(vdf > 0, vdf, 0.0)
    interp = RegularGridInterpolator(
        (ph_bin, th_bin, v_bin), vdf_clean,
        method='linear', bounds_error=False, fill_value=0.0)

    PH_fine, TH_fine, V_fine = np.meshgrid(ph_fine, th_fine, v_fine, indexing='ij')
    pts = np.column_stack([PH_fine.ravel(), TH_fine.ravel(), V_fine.ravel()])
    f_fine = interp(pts).reshape(n_ph * upsample, n_th * upsample, n_v * upsample)
    f_fine = np.clip(f_fine, 0, None)

    return ph_fine, th_fine, v_fine, f_fine


def resample_particles_from_vdf(V_para, V_perp1, V_perp2, vdf, vel,
                                 theta, phi, n_samples=20000,
                                 random_state=42,
                                 intra_bin='none',
                                 V_bulk_SRF=None,
                                 fac_basis=None,
                                 upsample=1):
    """
    Resample particles from a 3D VDF for GMM clustering in velocity space.

    Parameters
    ----------
    V_para, V_perp1, V_perp2 : ndarray, shape (N_az, N_el, N_en)
        Velocity components in field-aligned coordinates, in km/s.
    vdf : ndarray, shape (N_az, N_el, N_en)
        VDF values.
    vel : astropy Quantity or ndarray, shape (N_en,)
        Speed bin centres in km/s.
    theta : astropy Quantity or ndarray, shape (N_el,)
        Elevation angles in degrees.
    phi : astropy Quantity or ndarray, shape (N_az,)
        Azimuth angles in degrees.
    n_samples : int
        Total number of particles to sample.
    random_state : int
        Seed for reproducibility.
    intra_bin : str
        - 'none' (default): place all particles at the bin-centre velocity.
        - 'uniform': scatter uniformly within each bin in (v, θ, φ).
          When upsample > 1, bins are interpolated onto a finer grid first
          (see `upsample` parameter).
        - 'interp': probabilistic selection ∝ trilinearly interpolated f
          within each bin (M_try=5 candidates per particle).
    V_bulk_SRF : array-like, shape (3,), optional
        Proton bulk velocity in SRF [km/s].  Required for intra_bin != 'none'.
    fac_basis : tuple of arrays, optional
        FAC basis vectors (N, P, Q), each shape (3,), from
        Funcs.fieldAlignedCoordinates().  Required for intra_bin != 'none'.
    upsample : int
        (uniform mode only) Upsample factor per axis.  1 = use original PAS
        bins.  2 = 2× finer in φ, θ, v (8× cells).  Higher values make the
        "uniform within bin" approximation more accurate.

    Returns
    -------
    samples : ndarray, shape (n_samples, 3)
        Sampled particle velocities [V_para, V_perp1, V_perp2] in km/s.
    """
    rng = np.random.default_rng(random_state)

    # Only sample from non-zero VDF cells
    mask = vdf > 0
    n_nonzero = np.sum(mask)
    if n_nonzero == 0:
        raise ValueError("VDF is all zeros — nothing to sample.")

    idx_az, idx_el, idx_en = np.where(mask)

    f_vals = vdf[mask]

    # ---- Velocity-space volume element ----
    v_bin = vel.value if hasattr(vel, 'value') else np.asarray(vel)
    th_bin = theta.value if hasattr(theta, 'value') else np.asarray(theta)
    ph_bin = phi.value if hasattr(phi, 'value') else np.asarray(phi)

    # Energy bin widths: for log-spaced bins, Δv ∝ v
    dv = np.zeros_like(v_bin)
    dv[0] = v_bin[1] - v_bin[0]
    dv[-1] = v_bin[-1] - v_bin[-2]
    dv[1:-1] = (v_bin[2:] - v_bin[:-2]) / 2.0
    dv = np.abs(dv)  # guard against negative widths at boundaries

    # ---- Sampling weight ----
    # Weight by f × ΔV (velocity-space volume element), NOT by f alone.
    # The GMM fits a density in Cartesian velocity space, so the sampled
    # point DENSITY must be ∝ f; each cell must therefore receive particles
    # in proportion to its particle content f·ΔV, with
    #   ΔV = v² cosθ Δv Δθ Δφ   (Δv ∝ v on the log-spaced PAS grid → f v³ cosθ).
    # Weighting by f alone under-samples fast populations (alphas by ~2√2)
    # and drags each fitted mean inward by ~3σ²/v.  Full derivation:
    # Miscellany/Backup_Wrong_Resample/Principle/resampling_principle.pdf
    dtheta_w = _compute_bin_halfwidths(th_bin)
    dphi_w   = _compute_bin_halfwidths(ph_bin)
    weights = (f_vals
               * v_bin[idx_en]**2 * dv[idx_en]
               * np.cos(np.radians(th_bin[idx_el]))
               * dtheta_w[idx_el] * dphi_w[idx_az])
    weights = np.clip(weights, 0, None)
    if weights.sum() == 0:
        raise ValueError("All weights are zero — check your VDF and volume element.")

    probs = weights / weights.sum()

    # Sample cell indices
    flat_indices = np.arange(n_nonzero)
    sampled_cells = rng.choice(flat_indices, size=n_samples, p=probs)

    if intra_bin == 'none':
        # ---- Original behaviour: place particles at exact grid-point velocities ----
        V_para_flat = V_para[mask]
        V_perp1_flat = V_perp1[mask]
        V_perp2_flat = V_perp2[mask]

        samples = np.column_stack([
            V_para_flat[sampled_cells],
            V_perp1_flat[sampled_cells],
            V_perp2_flat[sampled_cells],
        ])

    elif intra_bin == 'uniform':
        # ---- Scatter particles uniformly within each bin's volume ----
        if V_bulk_SRF is None or fac_basis is None:
            raise ValueError("V_bulk_SRF and fac_basis are required for intra_bin != 'none'")

        N, P, Q = fac_basis

        if upsample > 1:
            # ---- Fine-grid mode: interpolate f onto a finer grid, then
            #     sample fine cells.  "uniform within bin" is essentially
            #     exact when the bin is tiny enough.
            ph_fine, th_fine, v_fine, f_fine = _build_fine_grid(
                vdf, ph_bin, th_bin, v_bin, upsample)

            # Fine-grid bin widths
            dv_fine   = _compute_bin_halfwidths(v_fine)
            dth_fine  = _compute_bin_halfwidths(th_fine)
            dph_fine  = _compute_bin_halfwidths(ph_fine)

            # Weight fine cells by f × ΔV_fine (same Jacobian as coarse path)
            mask_fine = f_fine > 0
            n_fine = np.sum(mask_fine)
            if n_fine == 0:
                raise ValueError("Fine-grid VDF is all zeros.")

            idx_ph, idx_th, idx_v = np.where(mask_fine)

            f_fine_flat = f_fine[mask_fine]
            w_fine = (f_fine_flat
                      * v_fine[idx_v]**2 * dv_fine[idx_v]
                      * np.cos(np.radians(th_fine[idx_th]))
                      * dth_fine[idx_th] * dph_fine[idx_ph])
            probs_fine = w_fine / w_fine.sum()
            flat_idx = np.arange(n_fine)
            sampled_fine = rng.choice(flat_idx, size=n_samples, p=probs_fine)

            unique_fine, counts_fine = np.unique(sampled_fine, return_counts=True)

            samples = np.empty((n_samples, 3))
            write_ptr = 0
            for cell_idx, count in zip(unique_fine, counts_fine):
                i_ph = idx_ph[cell_idx]
                i_th = idx_th[cell_idx]
                i_v  = idx_v[cell_idx]

                v_c  = v_fine[i_v]
                dv_h = dv_fine[i_v] / 2.0
                th_c = th_fine[i_th]
                dth_h = dth_fine[i_th] / 2.0
                ph_c = ph_fine[i_ph]
                dph_h = dph_fine[i_ph] / 2.0

                sub = _sample_within_bin(
                    v_c, dv_h, th_c, dth_h, ph_c, dph_h,
                    V_bulk_SRF, N, P, Q, count, rng)

                samples[write_ptr:write_ptr + count] = sub
                write_ptr += count

        else:
            # ---- Original coarse-grid mode ----
            dtheta = _compute_bin_halfwidths(th_bin)
            dphi   = _compute_bin_halfwidths(ph_bin)

            unique_cells, counts = np.unique(sampled_cells, return_counts=True)
            cell_to_count = dict(zip(unique_cells, counts))

            samples = np.empty((n_samples, 3))
            write_ptr = 0
            for cell_idx, count in cell_to_count.items():
                i_az = idx_az[cell_idx]
                i_el = idx_el[cell_idx]
                i_en = idx_en[cell_idx]

                v_c = v_bin[i_en]
                dv_half = dv[i_en] / 2.0
                th_c = th_bin[i_el]
                dth_half = dtheta[i_el] / 2.0
                ph_c = ph_bin[i_az]
                dph_half = dphi[i_az] / 2.0

                sub = _sample_within_bin(
                    v_c, dv_half, th_c, dth_half, ph_c, dph_half,
                    V_bulk_SRF, N, P, Q, count, rng)

                samples[write_ptr:write_ptr + count] = sub
                write_ptr += count

    elif intra_bin == 'interp':
        # ---- Scatter particles within bin ∝ interpolated f(v,θ,φ) ----
        if V_bulk_SRF is None or fac_basis is None:
            raise ValueError("V_bulk_SRF and fac_basis are required for intra_bin != 'none'")

        N, P, Q = fac_basis

        # Set up trilinear interpolator on the VDF grid
        # Grid order: (phi, theta, v) — matches VDF shape (N_az, N_el, N_en)
        vdf_clean = np.where(vdf > 0, vdf, 0.0)  # ensure non-negative
        interp = RegularGridInterpolator(
            (ph_bin, th_bin, v_bin), vdf_clean,
            method='linear', bounds_error=False, fill_value=0.0)

        # Compute angular bin half-widths
        dtheta = _compute_bin_halfwidths(th_bin)
        dphi   = _compute_bin_halfwidths(ph_bin)

        M_TRY = 5  # candidate positions per particle

        unique_cells, counts = np.unique(sampled_cells, return_counts=True)

        samples = np.empty((n_samples, 3))
        write_ptr = 0

        for cell_idx, count in zip(unique_cells, counts):
            i_az = idx_az[cell_idx]
            i_el = idx_el[cell_idx]
            i_en = idx_en[cell_idx]

            v_c = v_bin[i_en]
            dv_half = dv[i_en] / 2.0
            th_c = th_bin[i_el]
            dth_half = dtheta[i_el] / 2.0
            ph_c = ph_bin[i_az]
            dph_half = dphi[i_az] / 2.0

            # Generate all M_TRY candidates for all particles in this cell at once
            n_candidates = count * M_TRY
            v_cand = rng.uniform(v_c - dv_half, v_c + dv_half, size=n_candidates)
            th_cand = rng.uniform(th_c - dth_half, th_c + dth_half, size=n_candidates)
            ph_cand = rng.uniform(ph_c - dph_half, ph_c + dph_half, size=n_candidates)

            # Interpolate f at all candidate positions (vectorised)
            pts = np.column_stack([ph_cand, th_cand, v_cand])
            f_cand = interp(pts)
            f_cand = np.clip(f_cand, 0, None)

            # Reshape to (count, M_TRY) and select one candidate per particle
            f_reshaped = f_cand.reshape(count, M_TRY)
            row_sums = f_reshaped.sum(axis=1)

            # For rows where interpolated f is zero everywhere, fall back to uniform
            zero_rows = row_sums == 0
            f_reshaped[zero_rows, :] = 1.0  # uniform for zero rows

            # Select via cumulative-probability comparison
            cumsum = np.cumsum(f_reshaped, axis=1)
            rand_vals = rng.random(count)
            # For each row, find first column where cumsum >= rand * rowsum
            thresholds = rand_vals * cumsum[:, -1]
            chosen = np.argmax(cumsum >= thresholds[:, None], axis=1)

            # Extract chosen (v, th, ph) for each particle
            v_chosen   = v_cand.reshape(count, M_TRY)[np.arange(count), chosen]
            th_chosen  = th_cand.reshape(count, M_TRY)[np.arange(count), chosen]
            ph_chosen  = ph_cand.reshape(count, M_TRY)[np.arange(count), chosen]

            # Convert to FAC Cartesian
            th_rad = np.radians(th_chosen)
            ph_rad = np.radians(ph_chosen)

            vx = -v_chosen * np.cos(th_rad) * np.cos(ph_rad) - V_bulk_SRF[0]
            vy =  v_chosen * np.cos(th_rad) * np.sin(ph_rad) - V_bulk_SRF[1]
            vz = -v_chosen * np.sin(th_rad) - V_bulk_SRF[2]

            V_para_i  = vx * N[0] + vy * N[1] + vz * N[2]
            V_perp1_i = vx * P[0] + vy * P[1] + vz * P[2]
            V_perp2_i = vx * Q[0] + vy * Q[1] + vz * Q[2]

            samples[write_ptr:write_ptr + count] = np.column_stack(
                [V_para_i, V_perp1_i, V_perp2_i])
            write_ptr += count

    else:
        raise ValueError(f"Unknown intra_bin mode: {intra_bin!r}")

    return samples


def _compute_bin_halfwidths(arr):
    """
    Compute the half-width of each bin from an array of bin centres.

    Uses central differences for interior bins, one-sided at edges.
    Returns full bin widths (not half-widths).
    """
    widths = np.zeros_like(arr)
    widths[0] = arr[1] - arr[0]
    widths[-1] = arr[-1] - arr[-2]
    widths[1:-1] = (arr[2:] - arr[:-2]) / 2.0
    return np.abs(widths)


def _sample_within_bin(v_c, dv_half, th_deg, dth_half, ph_deg, dph_half,
                        V_bulk_SRF, N, P, Q, n_particles, rng):
    """
    Generate n_particles uniformly distributed within a spherical-coordinate
    bin and convert to FAC Cartesian [V_para, V_perp1, V_perp2].

    Parameters
    ----------
    v_c, dv_half : float
        Bin-centre speed [km/s] and half-width.
    th_deg, dth_half : float
        Bin-centre elevation [degrees] and half-width.
    ph_deg, dph_half : float
        Bin-centre azimuth [degrees] and half-width.
    V_bulk_SRF : array-like, shape (3,)
        Proton bulk velocity in SRF [km/s].
    N, P, Q : ndarray, each shape (3,)
        FAC basis vectors.
    n_particles : int
        Number of particles to generate.
    rng : numpy.random.Generator
        Random number generator.xfsort

    Returns
    -------
    samples : ndarray, shape (n_particles, 3)
        [V_para, V_perp1, V_perp2] in km/s.
    """
    # Sample uniformly within bin boundaries
    v_rand = rng.uniform(v_c - dv_half, v_c + dv_half, size=n_particles)
    th_rand_deg = rng.uniform(th_deg - dth_half, th_deg + dth_half, size=n_particles)
    ph_rand_deg = rng.uniform(ph_deg - dph_half, ph_deg + dph_half, size=n_particles)

    th_rand = np.radians(th_rand_deg)
    ph_rand = np.radians(ph_rand_deg)

    # Convert to SRF Cartesian
    vx = -v_rand * np.cos(th_rand) * np.cos(ph_rand) - V_bulk_SRF[0]
    vy =  v_rand * np.cos(th_rand) * np.sin(ph_rand) - V_bulk_SRF[1]
    vz = -v_rand * np.sin(th_rand) - V_bulk_SRF[2]

    # Rotate to FAC
    V_para  = vx * N[0] + vy * N[1] + vz * N[2]
    V_perp1 = vx * P[0] + vy * P[1] + vz * P[2]
    V_perp2 = vx * Q[0] + vy * Q[1] + vz * Q[2]

    return np.column_stack([V_para, V_perp1, V_perp2])


def bin_averaged_posterior(gmm, vdf_corrected,
                            vel, theta, phi,
                            V_bulk_SRF, fac_basis,
                            n_sub=10, random_state=None,
                            intra_bin='uniform',
                            upsample=1):
    """
    Compute bin-averaged GMM posteriors via Monte Carlo sub-sampling.

    For each non-zero VDF bin, evaluates the GMM posterior at sub-positions
    within the bin volume and averages.

    Two modes:
    - ``intra_bin='uniform'``: uniform average over sub-particles.
      When upsample > 1, uses a fine interpolated grid.
    - ``intra_bin='interp'``: f-weighted average, where sub-particle
      contributions are weighted by the trilinearly interpolated VDF
      value at each sub-position.

    Parameters
    ----------
    gmm : sklearn.mixture.GaussianMixture
        Fitted GMM with 3 components.
    vdf_corrected : ndarray, shape (N_az, N_el, N_en)
        Noise-cleaned VDF.
    vel : ndarray or Quantity, shape (N_en,)
        Speed bin centres in km/s.
    theta : ndarray or Quantity, shape (N_el,)
        Elevation angles in degrees.
    phi : ndarray or Quantity, shape (N_az,)
        Azimuth angles in degrees.
    V_bulk_SRF : array-like, shape (3,)
        Proton bulk velocity in SRF [km/s].
    fac_basis : tuple (N, P, Q)
        FAC basis vectors, each shape (3,).
    n_sub : int
        Number of Monte Carlo sub-particles per bin (unused when
        upsample > 1 and intra_bin='uniform').
    random_state : int, optional
        Seed.
    intra_bin : str
        'uniform' (default) or 'interp'.
    upsample : int
        (uniform mode only) Upsample factor for fine-grid evaluation.
        1 = use random sub-particles (original).  >1 = evaluate GMM at
        fine-grid cell centres and average within each coarse bin.

    Returns
    -------
    probas : ndarray, shape (n_nonzero, n_components)
        Bin-averaged posterior probabilities.
    """
    rng = np.random.default_rng(random_state)
    N, P, Q = fac_basis

    v_bin = vel.value if hasattr(vel, 'value') else np.asarray(vel)
    th_bin = theta.value if hasattr(theta, 'value') else np.asarray(theta)
    ph_bin = phi.value if hasattr(phi, 'value') else np.asarray(phi)

    mask = vdf_corrected > 0
    idx_az, idx_el, idx_en = np.where(mask)
    n_nonzero = len(idx_az)

    # ---- Fine-grid fast path (uniform + upsample > 1) ----
    if intra_bin == 'uniform' and upsample > 1:
        ph_fine, th_fine, v_fine, _ = _build_fine_grid(
            vdf_corrected, ph_bin, th_bin, v_bin, upsample)

        n_ph_f, n_th_f, n_v_f = len(ph_fine), len(th_fine), len(v_fine)

        # Map each fine cell to its parent coarse bin (nearest neighbour)
        # Vectorised: for each fine grid axis, find nearest coarse index
        i_ph_coarse = np.abs(ph_fine[:, None] - ph_bin[None, :]).argmin(axis=1)
        i_th_coarse = np.abs(th_fine[:, None] - th_bin[None, :]).argmin(axis=1)
        i_v_coarse  = np.abs(v_fine[:, None]  - v_bin[None, :]).argmin(axis=1)

        # Build a lookup: coarse bin (i_az, i_el, i_en) → list of fine cell FAC positions
        # First, compute FAC Cartesian for ALL fine cell centres (vectorised)
        PH_f, TH_f, V_f = np.meshgrid(ph_fine, th_fine, v_fine, indexing='ij')
        th_rad = np.radians(TH_f)
        ph_rad = np.radians(PH_f)

        vx = -V_f * np.cos(th_rad) * np.cos(ph_rad) - V_bulk_SRF[0]
        vy =  V_f * np.cos(th_rad) * np.sin(ph_rad) - V_bulk_SRF[1]
        vz = -V_f * np.sin(th_rad) - V_bulk_SRF[2]

        V_para_f  = vx * N[0] + vy * N[1] + vz * N[2]
        V_perp1_f = vx * P[0] + vy * P[1] + vz * P[2]
        V_perp2_f = vx * Q[0] + vy * Q[1] + vz * Q[2]

        X_fine = np.column_stack([
            V_para_f.ravel(), V_perp1_f.ravel(), V_perp2_f.ravel()])

        # Evaluate GMM on ALL fine cells at once
        probas_fine = gmm.predict_proba(X_fine)  # (N_fine, n_components)

        # Map fine→coarse: build a composite key for each fine cell
        i_ph_flat = np.broadcast_to(
            np.arange(n_ph_f)[:, None, None], (n_ph_f, n_th_f, n_v_f)).ravel()
        i_th_flat = np.broadcast_to(
            np.arange(n_th_f)[None, :, None], (n_ph_f, n_th_f, n_v_f)).ravel()
        i_v_flat  = np.broadcast_to(
            np.arange(n_v_f)[None, None, :], (n_ph_f, n_th_f, n_v_f)).ravel()

        coarse_ph = i_ph_coarse[i_ph_flat]
        coarse_th = i_th_coarse[i_th_flat]
        coarse_v  = i_v_coarse[i_v_flat]

        # Now average fine-cell posteriors within each coarse bin
        probas = np.empty((n_nonzero, gmm.n_components))
        for n in range(n_nonzero):
            in_bin = (coarse_ph == idx_az[n]) & (coarse_th == idx_el[n]) & (coarse_v == idx_en[n])
            if in_bin.any():
                probas[n] = probas_fine[in_bin].mean(axis=0)
            else:
                # Edge: bin contains no fine cells — evaluate at bin centre
                v_c = v_bin[idx_en[n]]
                th_c = th_bin[idx_el[n]]
                ph_c = ph_bin[idx_az[n]]
                sub = _sample_within_bin(
                    v_c, 0.0, th_c, 0.0, ph_c, 0.0,
                    V_bulk_SRF, N, P, Q, 1, rng)
                probas[n] = gmm.predict_proba(sub)[0]

        return probas

    # ---- Original per-bin loop (uniform upsample=1, or interp) ----
    dv = _compute_bin_halfwidths(v_bin)
    dtheta = _compute_bin_halfwidths(th_bin)
    dphi = _compute_bin_halfwidths(ph_bin)

    # Set up interpolator for f-weighted mode
    interp = None
    if intra_bin == 'interp':
        vdf_clean = np.where(vdf_corrected > 0, vdf_corrected, 0.0)
        interp = RegularGridInterpolator(
            (ph_bin, th_bin, v_bin), vdf_clean,
            method='linear', bounds_error=False, fill_value=0.0)

    probas = np.empty((n_nonzero, gmm.n_components))

    for n in range(n_nonzero):
        i_az = idx_az[n]
        i_el = idx_el[n]
        i_en = idx_en[n]

        v_c = v_bin[i_en]
        dv_half = dv[i_en] / 2.0
        th_c = th_bin[i_el]
        dth_half = dtheta[i_el] / 2.0
        ph_c = ph_bin[i_az]
        dph_half = dphi[i_az] / 2.0

        if intra_bin == 'interp' and interp is not None:
            # ---- f-weighted: generate in spherical, interpolate, weight ----
            v_rand = rng.uniform(v_c - dv_half, v_c + dv_half, size=n_sub)
            th_rand = rng.uniform(th_c - dth_half, th_c + dth_half, size=n_sub)
            ph_rand = rng.uniform(ph_c - dph_half, ph_c + dph_half, size=n_sub)

            pts = np.column_stack([ph_rand, th_rand, v_rand])
            f_weights = interp(pts)
            f_weights = np.clip(f_weights, 0, None)
            if f_weights.sum() > 0:
                f_norm = f_weights / f_weights.sum()
            else:
                f_norm = np.ones(n_sub) / n_sub

            # Convert to FAC
            th_rad = np.radians(th_rand)
            ph_rad = np.radians(ph_rand)
            vx = -v_rand * np.cos(th_rad) * np.cos(ph_rad) - V_bulk_SRF[0]
            vy =  v_rand * np.cos(th_rad) * np.sin(ph_rad) - V_bulk_SRF[1]
            vz = -v_rand * np.sin(th_rad) - V_bulk_SRF[2]
            V_para_i  = vx * N[0] + vy * N[1] + vz * N[2]
            V_perp1_i = vx * P[0] + vy * P[1] + vz * P[2]
            V_perp2_i = vx * Q[0] + vy * Q[1] + vz * Q[2]
            sub_samples = np.column_stack([V_para_i, V_perp1_i, V_perp2_i])

            sub_probas = gmm.predict_proba(sub_samples)
            probas[n] = np.average(sub_probas, axis=0, weights=f_norm)

        else:
            # ---- Uniform: use _sample_within_bin helper ----
            sub_samples = _sample_within_bin(
                v_c, dv_half, th_c, dth_half, ph_c, dph_half,
                V_bulk_SRF, N, P, Q, n_sub, rng)

            sub_probas = gmm.predict_proba(sub_samples)
            probas[n] = sub_probas.mean(axis=0)

    return probas


def _sort_components(f_all, gmm):
    """
    Identify and sort GMM components as [core, beam, alpha].

    Logic:
    - Core: component whose 1D reduced VDF has the HIGHEST peak value
      (proton core dominates the total VDF — easiest to identify)
    - Alpha: among the remaining two, component whose 1D peak is at larger |V|
      (alpha is shifted by sqrt(2) in the PAS velocity grid, so it appears
      farther in v than the proton beam)
    - Beam: the last one
    """

    # Determine alpha.
    n_component = len(f_all)

    # Scan from large speed to low speed, whichever starts with a higher speed is alpha.
    def find_first_nonzero_index(f_1d):
        return np.argmax(f_1d > 0)  # gives first non-zero index

    # Step 1: Reduce to 1D
    f0_1d = np.sum(f_all[0], axis=(0, 1))
    f1_1d = np.sum(f_all[1], axis=(0, 1))
    f2_1d = np.sum(f_all[2], axis=(0, 1))

    # Step 2: Get first non-zero indices
    a = find_first_nonzero_index(f0_1d)
    b = find_first_nonzero_index(f1_1d)
    c = find_first_nonzero_index(f2_1d)

    # Step 3: Find max index shared across all
    idx = np.max([a, b, c])

    # Step 4: Compare values at this index
    values_at_idx = [f0_1d[idx], f1_1d[idx], f2_1d[idx]]
    alpha_index = int(np.argmax(values_at_idx))
    
    # Among the remaining two, determine core and beam based on peak value
    remaining_indices = [i for i in range(n_component) if i != alpha_index]
    peak_values = [np.max(f_all[i]) for i in remaining_indices]
    if peak_values[0] >= peak_values[1]:
        core_index = remaining_indices[0]
        beam_index = remaining_indices[1]
    else:
        core_index = remaining_indices[1]
        beam_index = remaining_indices[0]

    # Reorder f_all accordingly: [core, beam, alpha]
    sort_indices = [alpha_index, beam_index, core_index]

    # Sort covariance using the same indices
    f_all_sorted = [f_all[i] for i in sort_indices]
    covariance_sorted = [gmm.covariances_[i] for i in sort_indices]
    means_sorted = [gmm.means_[i] for i in sort_indices]
    weights_sorted = [gmm.weights_[i] for i in sort_indices]

    return f_all_sorted, [means_sorted, covariance_sorted, weights_sorted]


def _compute_alpha_perp_metric(f_sorted, vel, theta, phi,
                                measured_time, magF_SRF,
                                Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz):
    """
    Compute the perpendicular coupling metric for a GMM fit.

    Metric = |Vα,⊥ − Vp,⊥| / |Vp,‖|

    Uses the full bulk velocities of the separated proton and alpha VDFs
    (via SolarWindPack), exactly matching the old tutorial's Flag_Vperp.

    Returns (delta_V_perp_norm, f_ratio).
    """
    try:
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from SolarWindPack import SolarWindParticle, cal_bulk_velocity_Spherical
        from Funcs import rotateVectorIntoFieldAligned
    except ImportError:
        # Can't import SolarWindPack/Funcs — let caller fall back
        raise

    f_alpha, f_beam, f_core = f_sorted

    v_vals = vel.value if hasattr(vel, 'value') else np.asarray(vel)
    th_vals = theta.value if hasattr(theta, 'value') else np.asarray(theta)
    ph_vals = phi.value if hasattr(phi, 'value') else np.asarray(phi)

    Protons = SolarWindParticle(
        'proton', time=measured_time, magfield=magF_SRF,
        grid=[th_vals, ph_vals, v_vals * 1e3], coord_type='Spherical')
    Protons.set_vdf(f_core, 'core')
    Protons.set_vdf(f_beam, 'beam')

    Alphas = SolarWindParticle(
        'alpha', time=measured_time, magfield=magF_SRF,
        grid=[th_vals, ph_vals, v_vals * 1e3 / np.sqrt(2)],
        coord_type='Spherical')
    Alphas.set_vdf(f_alpha * 4)

    Vproton = cal_bulk_velocity_Spherical(Protons) / 1e3
    Valpha = cal_bulk_velocity_Spherical(Alphas) / 1e3

    Vp_BA = rotateVectorIntoFieldAligned(
        Vproton[0], Vproton[1], Vproton[2],
        Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    Va_BA = rotateVectorIntoFieldAligned(
        Valpha[0], Valpha[1], Valpha[2],
        Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)

    Vp_para = Vp_BA[0]
    Vp_perp = np.sqrt(Vp_BA[1]**2 + Vp_BA[2]**2)
    Va_perp = np.sqrt(Va_BA[1]**2 + Va_BA[2]**2)

    if abs(Vp_para) < 1.0:
        return np.inf, np.inf

    delta_V_perp_norm = abs(Va_perp - Vp_perp) / abs(Vp_para)

    f_alpha_max = np.max(f_alpha)
    f_proton_max = np.max(f_core + f_beam)
    f_ratio = f_alpha_max / f_proton_max if f_proton_max > 0 else np.inf

    return delta_V_perp_norm, f_ratio


def _compute_alpha_perp_metric_from_means(f_sorted, gmm_info):
    """
    Fallback metric using only GMM component means (no SolarWindPack).

    Less accurate than the bulk-velocity version but works standalone.
    """
    if gmm_info is None:
        # Need at least the sorted component means
        return np.inf, np.inf

    means = gmm_info[0]  # [core, beam, alpha] means in FAC
    mu_core = np.asarray(means[0])
    mu_alpha = np.asarray(means[2])

    Vp_para = mu_core[0]
    if abs(Vp_para) < 1.0:
        return np.inf, np.inf

    Vp_perp = np.sqrt(mu_core[1]**2 + mu_core[2]**2)
    Va_perp = np.sqrt(mu_alpha[1]**2 + mu_alpha[2]**2)
    delta_V_perp_norm = abs(Va_perp - Vp_perp) / abs(Vp_para)

    f_alpha_max = np.max(f_sorted[2])
    f_proton_max = np.max(f_sorted[0] + f_sorted[1])
    f_ratio = f_alpha_max / f_proton_max if f_proton_max > 0 else np.inf

    return delta_V_perp_norm, f_ratio


def cal_GMM_resampled(V_para, V_perp1, V_perp2, vdf_corrected,
                       vel, theta, phi,
                       initial_means_3d, n_component=3,
                       co_type='auto', n_samples=20000,
                       random_state=42,
                       measured_time=None, magF_SRF=None,
                       Nx=None, Ny=None, Nz=None,
                       Px=None, Py=None, Pz=None,
                       Qx=None, Qy=None, Qz=None,
                       intra_bin='none',
                       V_bulk_SRF=None,
                       bin_avg_posterior=False,
                       n_sub=10,
                       upsample=1):
    """
    Run GMM on resampled particles in pure 3D velocity space.

    Particles are sampled from the VDF with probability proportional to
    f × d³v (volume element), then GMM is fit in pure (V_para, V_perp1, V_perp2)
    coordinates.  The fitted posteriors are applied back to the original
    grid to partition the VDF into core, beam, and alpha components.

    Parameters
    ----------
    V_para, V_perp1, V_perp2 : ndarray, shape (N_az, N_el, N_en)
        Velocity components in field-aligned coordinates (km/s).
    vdf_corrected : ndarray, shape (N_az, N_el, N_en)
        Noise-cleaned VDF (s³/m⁶ or equivalent).
    vel, theta, phi : ndarray or Quantity
        PAS grid coordinates.
    initial_means_3d : ndarray, shape (n_component, 3)
        Initial mean velocities for each component in [V_para, V_perp1, V_perp2].
    n_component : int
        Number of Gaussian components (3: core, beam, alpha).
    co_type : str
        GMM covariance type.  One of:
        - 'spherical', 'diag', 'tied', 'full' — use that type.
        - 'auto' — try all types, pick the best using a physics-based
          criterion: smallest |Vα,⊥ − Vp,⊥| / |Vp,‖| (alpha and proton
          should be well-coupled perpendicular to B), after filtering out
          fits where f_alpha_max / f_proton_max > 0.2.
          When using 'auto', provide measured_time, magF_SRF, and
          Nx..Qz for the full bulk-velocity metric.
    n_samples : int
        Number of particles to resample.
    random_state : int
        Seed.
    measured_time : datetime, optional
        Measurement time for SolarWindParticle. Needed for co_type='auto'.
    magF_SRF : array-like, optional
        Magnetic field in SRF. Needed for co_type='auto'.
    Nx..Qz : float, optional
        FAC basis vectors from Funcs.fieldAlignedCoordinates().
    intra_bin : str
        - 'none' (default): place particles at bin-centre velocity.
        - 'uniform': scatter uniformly within each bin in spherical coords.
    V_bulk_SRF : array-like, shape (3,), optional
        Proton bulk velocity in SRF [km/s].  Needed for intra_bin != 'none'.
    bin_avg_posterior : bool
        If True, use Monte Carlo bin-averaging for the posterior
        (defines the partitioned VDF via sub-particle evaluation).
    n_sub : int
        Number of sub-particles per bin for bin-averaged posterior.
    upsample : int
        (uniform mode only) Upsample factor for fine-grid resampling
        and posterior evaluation. 1 = use original PAS bins. 2 = 2×
        finer in φ, θ, v (8× cells).

    Returns
    -------
    f_all_sorted : list of ndarray
        Partitioned VDFs [f_core, f_beam, f_alpha], each shape (N_az, N_el, N_en).
    gmm_info : list
        [means, covariances, weights] of the fitted GMM.
    probas : ndarray
        Posterior probabilities for each non-zero grid point.
    chosen_type : str
        The covariance type actually used (only meaningful when co_type='auto').
    """
    # ---- Build FAC basis tuple for convenience ----
    fac_basis = None
    if all(x is not None for x in [Nx, Px]):
        fac_basis = (
            np.array([Nx, Ny, Nz]),
            np.array([Px, Py, Pz]),
            np.array([Qx, Qy, Qz]),
        )

    # ---- Handle 'auto' mode ----
    if co_type == 'auto':
        return _cal_GMM_auto(V_para, V_perp1, V_perp2, vdf_corrected,
                             vel, theta, phi,
                             initial_means_3d, n_component,
                             n_samples, random_state,
                             measured_time, magF_SRF,
                             Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz,
                             intra_bin, V_bulk_SRF, fac_basis,
                             bin_avg_posterior, n_sub,
                             upsample)

    # ---- Step 1: Resample particles ----
    samples = resample_particles_from_vdf(
        V_para, V_perp1, V_perp2, vdf_corrected,
        vel, theta, phi,
        n_samples=n_samples,
        random_state=random_state,
        intra_bin=intra_bin,
        V_bulk_SRF=V_bulk_SRF,
        fac_basis=fac_basis,
        upsample=upsample,
    )

    # ---- Step 2: Fit GMM in 3D velocity space ---print("Vp: ", Vproton)
    # Use KMeans on sampled particles to get initial weights
    kmeans = KMeans(n_clusters=n_component, random_state=random_state,
                    n_init=10).fit(samples)
    labels = kmeans.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    initial_weights = counts / len(labels)

    gmm = GaussianMixture(
        n_components=n_component,
        random_state=random_state,
        covariance_type=co_type,
        means_init=initial_means_3d,
        weights_init=initial_weights,
    ).fit(samples)

    # ---- Step 3: Predict posteriors for ORIGINAL grid points ----
    non_zero_idx = np.where(vdf_corrected > 0)

    if bin_avg_posterior and fac_basis is not None and V_bulk_SRF is not None:
        probas = bin_averaged_posterior(
            gmm, vdf_corrected, vel, theta, phi,
            V_bulk_SRF, fac_basis,
            n_sub=n_sub, random_state=random_state,
            intra_bin=intra_bin,
            upsample=upsample)
    else:
        non_zero_vpara = V_para[non_zero_idx]
        non_zero_vperp1 = V_perp1[non_zero_idx]
        non_zero_vperp2 = V_perp2[non_zero_idx]

        X_grid = np.column_stack([non_zero_vpara, non_zero_vperp1, non_zero_vperp2])
        probas = gmm.predict_proba(X_grid)

    # ---- Step 4: Partition the VDF ----
    f_all = [np.zeros_like(vdf_corrected) for _ in range(n_component)]
    for i in range(n_component):
        f_all[i][non_zero_idx] = probas[:, i] * vdf_corrected[non_zero_idx]

    # ---- Step 5: Identify which component is which ----
    f_all_sorted, gmm_info = _sort_components(f_all, gmm)

    return f_all_sorted, gmm_info, probas, co_type


def _cal_GMM_auto(V_para, V_perp1, V_perp2, vdf_corrected,
                   vel, theta, phi,
                   initial_means_3d, n_component,
                   n_samples, random_state,
                   measured_time, magF_SRF,
                   Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz,
                   intra_bin='none', V_bulk_SRF=None,
                   fac_basis=None,
                   bin_avg_posterior=False, n_sub=10,
                   upsample=1):
    """
    Run all covariance types and pick the best using the physics criterion:

    1. Filter: f_alpha_max / f_proton_max ≤ 0.2
       (if alpha VDF peak is too large relative to proton, the fit is unphysical)
    2. Pick: smallest |Vα,⊥ − Vp,⊥| / |Vp,‖|
       (alpha should be well-coupled to protons perpendicular to B)

    Uses the full bulk-velocity computation (via SolarWindPack) matching
    the old tutorial exactly when the extra parameters are provided.
    Falls back to GMM-means heuristic otherwise.
    """
    candidate_types = ['spherical', 'diag', 'tied', 'full']

    # Build metric kwargs if we have the full set of parameters
    metric_kwargs = {}
    if all(x is not None for x in [measured_time, magF_SRF, Nx]):
        metric_kwargs = {
            'vel': vel, 'theta': theta, 'phi': phi,
            'measured_time': measured_time, 'magF_SRF': magF_SRF,
            'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
            'Px': Px, 'Py': Py, 'Pz': Pz,
            'Qx': Qx, 'Qy': Qy, 'Qz': Qz,
        }

    best_result = None
    best_metric = np.inf
    best_type = None

    for co_type in candidate_types:
        try:
            f_sorted, gmm_info, probas, _ = cal_GMM_resampled(
                V_para, V_perp1, V_perp2, vdf_corrected,
                vel, theta, phi,
                initial_means_3d, n_component=n_component,
                co_type=co_type, n_samples=n_samples,
                random_state=random_state,
                intra_bin=intra_bin, V_bulk_SRF=V_bulk_SRF,
                Nx=Nx, Ny=Ny, Nz=Nz, Px=Px, Py=Py, Pz=Pz, Qx=Qx, Qy=Qy, Qz=Qz,
                bin_avg_posterior=bin_avg_posterior, n_sub=n_sub,
                upsample=upsample,
            )

            try:
                if metric_kwargs:
                    delta_V_perp, f_ratio = _compute_alpha_perp_metric(
                        f_sorted, **metric_kwargs)
                else:
                    delta_V_perp, f_ratio = _compute_alpha_perp_metric_from_means(
                        f_sorted, gmm_info)
            except Exception:
                # If bulk-velocity computation fails, fall back to GMM means
                delta_V_perp, f_ratio = _compute_alpha_perp_metric_from_means(
                    f_sorted, gmm_info)

            # Filter: alpha VDF peak shouldn't exceed 20% of proton peak
            if f_ratio > 0.2:
                continue

            if delta_V_perp < best_metric:
                best_metric = delta_V_perp
                best_result = (f_sorted, gmm_info, probas)
                best_type = co_type

        except Exception:
            continue

    if best_result is None:
        # Fallback: if all failed the filter, pick the one with smallest metric
        for co_type in candidate_types:
            try:
                f_sorted, gmm_info, probas, _ = cal_GMM_resampled(
                    V_para, V_perp1, V_perp2, vdf_corrected,
                    vel, theta, phi,
                    initial_means_3d, n_component=n_component,
                    co_type=co_type, n_samples=n_samples,
                    random_state=random_state,
                    intra_bin=intra_bin, V_bulk_SRF=V_bulk_SRF,
                    Nx=Nx, Ny=Ny, Nz=Nz, Px=Px, Py=Py, Pz=Pz, Qx=Qx, Qy=Qy, Qz=Qz,
                    bin_avg_posterior=bin_avg_posterior, n_sub=n_sub,
                )
                try:
                    if metric_kwargs:
                        delta_V_perp, _ = _compute_alpha_perp_metric(
                            f_sorted, **metric_kwargs)
                    else:
                        delta_V_perp, _ = _compute_alpha_perp_metric_from_means(
                            f_sorted, gmm_info)
                except Exception:
                    delta_V_perp, _ = _compute_alpha_perp_metric_from_means(
                        f_sorted, gmm_info)
                if delta_V_perp < best_metric:
                    best_metric = delta_V_perp
                    best_result = (f_sorted, gmm_info, probas)
                    best_type = co_type
            except Exception:
                continue

    if best_result is None:
        raise RuntimeError("All GMM covariance types failed.")

    f_sorted, gmm_info, probas = best_result
    return f_sorted, gmm_info, probas, best_type


def run_all_covariance_types(V_para, V_perp1, V_perp2, vdf_corrected,
                              vel, theta, phi,
                              initial_means_3d, n_component=3,
                              n_samples=50000, random_state=42,
                              intra_bin='none', V_bulk_SRF=None,
                              Nx=None, Ny=None, Nz=None,
                              Px=None, Py=None, Pz=None,
                              Qx=None, Qy=None, Qz=None,
                              bin_avg_posterior=False, n_sub=10,
                              upsample=1):
    """
    Try all GMM covariance types with resampling and return results.

    Returns
    -------
    results : dict
        Keys: 'full', 'diag', 'tied', 'spherical'
        Values: (f_all_sorted, gmm_info, probas)
    """
    results = {}
    for co_type in ['full', 'diag', 'tied', 'spherical']:
        try:
            f_sorted, gmm_info, probas, _ = cal_GMM_resampled(
                V_para, V_perp1, V_perp2, vdf_corrected,
                vel, theta, phi,
                initial_means_3d, n_component=n_component,
                co_type=co_type, n_samples=n_samples,
                random_state=random_state,
                intra_bin=intra_bin, V_bulk_SRF=V_bulk_SRF,
                Nx=Nx, Ny=Ny, Nz=Nz, Px=Px, Py=Py, Pz=Pz, Qx=Qx, Qy=Qy, Qz=Qz,
                bin_avg_posterior=bin_avg_posterior, n_sub=n_sub,
                upsample=upsample,
            )
            results[co_type] = (f_sorted, gmm_info, probas)
        except Exception as e:
            print(f"  [{co_type}] failed: {e}")
            results[co_type] = None
    return results
