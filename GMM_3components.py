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

The results are stored in objects and saved.
"""

# import functions, including the ones that I wrote.
import os
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

from Funcs import *
from SolarWindPack import *

def FindIndexinInterval(tstart, tend, epoch):
    # Given the epoch of the data, find all the indexes within the given interval.
    if tstart > tend:
        raise ValueError("tstart should be smaller than tend.")
    result = [[idx, epoch[idx]] for idx in range(len(epoch)) if tstart <= epoch[idx] <= tend]
    return result

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

def cal_GMM(V_para, V_perp1, V_perp2, vdf_corrected, co_type, initial_means, n_component):
    # Function for calculating GMM.
    non_zero_idx = np.where(vdf_corrected > 0)
    non_zero_vdf = vdf_corrected[non_zero_idx]
    non_zero_vpara = V_para[non_zero_idx]
    non_zero_vperp1 = V_perp1[non_zero_idx]
    non_zero_vperp2 = V_perp2[non_zero_idx]
    non_zero_magni = np.sqrt(non_zero_vpara**2 + non_zero_vperp1**2 + non_zero_vperp2**2)
    
    X = np.column_stack([non_zero_vpara, non_zero_vperp1, non_zero_vperp2, non_zero_magni, non_zero_vdf])

    # Use KMeans to determine initial weights
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    labels = kmeans.labels_
    # Compute the initial weights as the fraction of samples in each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    initial_weights = counts / len(labels)

    gmm_kwargs = {
        "n_components": n_component,
        "random_state": 42,
        "covariance_type": co_type,
        "means_init": initial_means,
        "weights_init": initial_weights,
    }

    gmm = GaussianMixture(**gmm_kwargs).fit(X)
    probas = gmm.predict_proba(X)

    f_all = [np.zeros_like(vdf_corrected) for _ in range(n_component)]

    for i in range(n_component):
        f_all[i][non_zero_idx] = probas[:, i] * vdf_corrected[non_zero_idx]

    # Determine alpha.
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

    # Step 3: Find smallest index shared across all
    idx = min(a, b, c)

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

    # Set small values to 0 in f_all.
    for f in f_all_sorted:
        f[f < 1e-14] = 0

    return f_all_sorted, [means_sorted, covariance_sorted, weights_sorted], probas



def all_process(idx_time, Protons_initial, Alphas_initial, vdf_cdffile, grnd_cdffile, mag_cdffile, one_particle_noise_level, result_path):
    """
    A function with all the processes, with figures plotted.
    """
    tsliceindex_vdf = idx_time[0]
    tslice_vdf = idx_time[1]

    # Get all the initial values.
    epoch_vdf = vdf_cdffile['Epoch'][...]
    epoch_mag = mag_cdffile['EPOCH'][...]

    # The magnetic field is obtained by calculating the average of the 1-second magnetic field data around the time slice.
    tslice_vdf_start = epoch_vdf[tsliceindex_vdf] - timedelta(seconds=0.5)
    tslice_vdf_end = epoch_vdf[tsliceindex_vdf] + timedelta(seconds=0.5)
    tsliceindex_mag_start = bisect.bisect_left(epoch_mag, tslice_vdf_start)
    tsliceindex_mag_end = bisect.bisect_left(epoch_mag, tslice_vdf_end)
    print('Time for MAG: ', epoch_mag[tsliceindex_mag_start], " to ", epoch_mag[tsliceindex_mag_end])
    print("Time index for MAG: ", tsliceindex_mag_start, " to ", tsliceindex_mag_end)

    # Read data from SO product.
    V_bulk_SRF = grnd_cdffile['V_SRF'][tsliceindex_vdf]
    vdf = vdf_cdffile['vdf'][tsliceindex_vdf]

    qp = 1.60217662e-19 # C
    mp = 1.6726219e-27 # kg

    vel = np.sqrt(2 * vdf_cdffile['Energy'][...] * qp / mp) / 1e3 * (u.km / u.s) # in km/s
    theta = vdf_cdffile['Elevation'][...] * u.deg # in deg
    phi = vdf_cdffile['Azimuth'][...] * u.deg # in deg

    magF_SRF = mag_cdffile['B_SRF'][tsliceindex_mag_start:tsliceindex_mag_end].mean(axis=0)

    # Calculate the local Alfven speed to set initial means.
    B_magnitude = np.sqrt(np.sum(magF_SRF**2)) * 1e-9  #T
    density = grnd_cdffile['N'][tsliceindex_vdf] * 1e6 # m^-3
    mu0 = 4 * np.pi * 1e-7 # N/A^2
    mp = 1.67*1e-27 # kg
    VA = B_magnitude / np.sqrt(mu0 * density * mp) / 1000.0 #  km/s

    # Any anode with measurement below the noise level is considered as noise, and removed.
    vdf_corrected = vdf.copy()
    vdf_corrected[vdf_corrected <= one_particle_noise_level] = 0

    print("Before one-particel noise level: ", np.count_nonzero(vdf))
    print("After one-particel noise level: ", np.count_nonzero(vdf_corrected))

    # Let's first get the f and (vx, vy, vz) grid points.
    vx = np.zeros((11, 9, 96))
    vy = np.zeros((11, 9, 96))
    vz = np.zeros((11, 9, 96))

    # Calculate in SRF coordinate.
    for i in range(11):
        for j in range(9):
            vx[i, j, :] = - vel * np.cos(theta[j]) * np.cos(phi[i]) - V_bulk_SRF[0] * u.km / u.s
            vy[i, j, :] = vel * np.cos(theta[j]) * np.sin(phi[i]) - V_bulk_SRF[1] * u.km / u.s
            vz[i, j, :] = - vel * np.sin(theta[j]) - V_bulk_SRF[2] * u.km / u.s

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

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, label='Corrected VDF')
    ax.scatter(x, y, s=20, color='red')
    ax.set_ylim(-12, -5)
    for idx, (xi, yi) in enumerate(zip(x.value, y)):
        ax.annotate(str(np.where(mask)[0][idx]), (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
    ax.set_xlabel('Vel [km/s]')
    ax.set_ylabel('log10(VDF)')
    plt.savefig(result_path + '/Corrected_VDF_1D.png')
    plt.close()

    # Correct again. Remove the points who have no neighboring points.
    def remove_noise(vdf_corrected):
        """
        Set to zero any point in vdf_corrected (shape: 11, 9, 96)
        that does not have at least one neighbor with a positive value.
        
        Parameters:
            vdf_corrected (ndarray): 3D array of shape (11, 9, 96)
        
        Returns:
            ndarray: Cleaned array with noise points set to zero
        """
        # Initialize a mask of False
        neighbor_mask = np.zeros_like(vdf_corrected, dtype=bool)

        # Axis 0 neighbors
        neighbor_mask[:-1, :, :] |= vdf_corrected[1:, :, :] > 0
        neighbor_mask[1:, :, :] |= vdf_corrected[:-1, :, :] > 0

        # Axis 1 neighbors
        neighbor_mask[:, :-1, :] |= vdf_corrected[:, 1:, :] > 0
        neighbor_mask[:, 1:, :] |= vdf_corrected[:, :-1, :] > 0

        # Axis 2 neighbors
        neighbor_mask[:, :, :-1] |= vdf_corrected[:, :, 1:] > 0
        neighbor_mask[:, :, 1:] |= vdf_corrected[:, :, :-1] > 0

        # Zero out points with no positive neighbor
        cleaned_vdf = vdf_corrected.copy()
        cleaned_vdf[~neighbor_mask] = 0

        return cleaned_vdf

    vdf_corrected = remove_noise(vdf_corrected)

    # Plot to see the 1D vdf.
    y = log10_vdf(np.sum(vdf_corrected, axis=(0, 1)))
    mask = y != 0
    x = vel[mask]
    y = y[mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, label='Corrected VDF')
    ax.scatter(x, y, s=20, color='red')
    ax.set_ylim(-12, -5)
    for idx, (xi, yi) in enumerate(zip(x.value, y)):
        ax.annotate(str(np.where(mask)[0][idx]), (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
    ax.set_xlabel('Vel [km/s]')
    ax.set_ylabel('log10(VDF)')
    plt.savefig(result_path + '/Corrected_VDF_Cleaned.png')
    plt.close()

    # Get the dividing index bease on the previous vdfs.
    Pvdf_sum = np.sum(Protons_initial.get_vdf(), axis=(0, 1))
    Avdf_sum = np.sum(Alphas_initial.get_vdf(), axis=(0, 1))

    for i in range(len(Pvdf_sum)):
        if Avdf_sum[i] < Pvdf_sum[i]:
            dividing_idx = i
            break

    # Get the initial values for GMM.
    u_proton_core = cal_bulk_velocity_Spherical(Protons_initial, component='core')
    u_proton_core_Baligned = rotateVectorIntoFieldAligned(u_proton_core[0], u_proton_core[1], u_proton_core[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    u_proton_beam = cal_bulk_velocity_Spherical(Protons_initial, component='beam')
    u_proton_beam_Baligned = rotateVectorIntoFieldAligned(u_proton_beam[0], u_proton_beam[1], u_proton_beam[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    
    # For alphas, we need to multiply the velocity by sqrt(2) to let GMM separate in the PAS reference.
    u_alpha = cal_bulk_velocity_Spherical(Alphas_initial) * np.sqrt(2)
    u_alpha_Baligned = rotateVectorIntoFieldAligned(u_alpha[0], u_alpha[1], u_alpha[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)

    diff_beam_core = (np.array(u_proton_beam_Baligned) - np.array(u_proton_core_Baligned)) / 1e3
    diff_alpha_core = (np.array(u_alpha_Baligned) - np.array(u_proton_core_Baligned)) / 1e3

    # Initial means
    # Here are two choices, if the beam is really not separate enough, I strongly suggest to use VA as initial input for beam.
    #initial_means = np.array([
    #    [0, 0, 0, 0, Protons_initial.get_vdf(component='core').max()],
    #    np.append(np.append(diff_beam_core, np.linalg.norm(diff_beam_core)), Protons_initial.get_vdf(component='beam').max()),
    #    np.append(np.append(diff_alpha_core, np.linalg.norm(diff_alpha_core)), Alphas_initial.get_vdf().max())
    #])

    initial_means = np.array([
        [0, 0, 0, 0, Protons_initial.get_vdf().max()], 
        [VA, 0, 0, VA, Protons_initial.get_vdf().max() / 10.0],
        np.append(np.append(diff_alpha_core, np.linalg.norm(diff_alpha_core)), Alphas_initial.get_vdf().max())
    ])

    n_component = 3
    f_full, dist_paras_full, probas_full = cal_GMM(V_para, V_perp1, V_perp2, vdf_corrected, 'full', initial_means, n_component)
    f_alpha_full, f_beam_full, f_core_full = f_full
    f_diag, dist_paras_diag, probas_diag = cal_GMM(V_para, V_perp1, V_perp2, vdf_corrected, 'diag', initial_means, n_component)
    f_alpha_diag, f_beam_diag, f_core_diag = f_diag
    f_spherical, dist_paras_spherical, probas_spherical = cal_GMM(V_para, V_perp1, V_perp2, vdf_corrected, 'spherical', initial_means, n_component)
    f_alpha_spherical, f_beam_spherical, f_core_spherical = f_spherical
    f_tied, dist_paras_tied, probas_tied= cal_GMM(V_para, V_perp1, V_perp2, vdf_corrected, 'tied', initial_means, n_component)
    f_alpha_tied, f_beam_tied, f_core_tied = f_tied

    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    # Full 
    plot_one(ax[0, 0], ax[1, 0], vel, vdf_corrected, f_core_full, f_beam_full, f_alpha_full, 'Full')
    # diag
    plot_one(ax[0, 1], ax[1, 1], vel, vdf_corrected, f_core_diag, f_beam_diag, f_alpha_diag, 'Diag')
    # tied
    plot_one(ax[0, 2], ax[1, 2], vel, vdf_corrected, f_core_tied, f_beam_tied, f_alpha_tied, 'Tied')
    # spherical
    plot_one(ax[0, 3], ax[1, 3], vel, vdf_corrected, f_core_spherical, f_beam_spherical, f_alpha_spherical, 'Spherical')
    plt.savefig(result_path + '/GMM_all.png')
    plt.close() 

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
    print("Angle between drift and B (Diag): ", Theta_ProtonAlpha_Diag)

    Theta_ProtonAlpha_Spherical = angle_between_vectors(magF_SRF, VdriftSpherical) * 180 / np.pi
    print("Angle between drift and B (Spherical): ", Theta_ProtonAlpha_Spherical)

    Theta_ProtonAlpha_Tied = angle_between_vectors(magF_SRF, VdriftTied) * 180 / np.pi
    print("Angle between drift and B (Tied): ", Theta_ProtonAlpha_Tied)

    # the ratio of max(Alpha VDF) / max(Proton VDF)
    # Sometimes when the separation is doing nonsense, this helps rule them out
    VDF_Amax_Pmax_tied = np.max(Alphas_tied.get_vdf()) / np.max(Protons_tied.get_vdf())
    print("VDF_Amax / VDF_Pmax (tied): ", VDF_Amax_Pmax_tied)   
    VDF_Amax_Pmax_spherical = np.max(Alphas_spherical.get_vdf()) / np.max(Protons_spherical.get_vdf())
    print("VDF_Amax / VDF_Pmax (spherical): ", VDF_Amax_Pmax_spherical)   
    VDF_Amax_Pmax_diag = np.max(Alphas_diag.get_vdf()) / np.max(Protons_diag.get_vdf())
    print("VDF_Amax / VDF_Pmax (diag): ", VDF_Amax_Pmax_diag)

    def alignment_score(theta_deg):
        """Return distance to best alignment (0° or 180°)."""
        theta = np.abs(theta_deg)
        return min(theta, np.abs(180 - theta))

    # Put all angles and corresponding data in a list of tuples
    options = [
        (alignment_score(Theta_ProtonAlpha_Diag), "Diag", Protons_diag, Alphas_diag, VDF_Amax_Pmax_diag),
        (alignment_score(Theta_ProtonAlpha_Spherical), "Spherical", Protons_spherical, Alphas_spherical, VDF_Amax_Pmax_spherical),
        (alignment_score(Theta_ProtonAlpha_Tied), "Tied", Protons_tied, Alphas_tied, VDF_Amax_Pmax_tied)
    ]

    # 1) Remove options with a too large alpha VDF peak
    # 2) If Alpha_VDF peak is larger than 25% Proton_VDF peak, not sensible! Remove!
    VDF_Amax_Pmax_threshold = 0.25
    valid_options = [
        opt for opt in options
        if (opt[4] is not None) and np.isfinite(opt[4]) and (opt[4] <= VDF_Amax_Pmax_threshold)
    ]

    # The one with the smallest theta is the best.
    if len(valid_options) == 0:
        print("No valid separation found based on criteria.")
        best_option = min(options, key=lambda x: x[0])
    else:
        best_option = min(valid_options, key=lambda x: x[0])

    # Unpack the result
    best_theta, which_one, Protons_current, Alphas_current, _= best_option

    #print(f"{which_one} is better.")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plot_one(ax[0], ax[1], vel, vdf_corrected, Protons_current.get_vdf(component='core'), Protons_current.get_vdf(component='beam'), Alphas_current.get_vdf() / 4, 'Final')
    plt.savefig(result_path + '/Final_Result.png')
    plt.close()

    # Moments.
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
    if np.abs(best_theta) > 40:
        print("Separation failed, use the previous separation.")
        Protons_current = Protons_initial
        Alphas_current = Alphas_initial

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
        "Np": (Npcore + Npbeam) / 1e6,
        "Nalpha_over_Np": Nalpha / (Npcore + Npbeam),
        "VpcorePara": VpcorePara / 1e3,
        "VpcorePerp": VpcorePerp / 1e3,
        "VpbeamPara": VpbeamPara / 1e3,
        "VpbeamPerp": VpbeamPerp / 1e3,
        "VprotonPara": VprotonPara / 1e3,
        "VprotonPerp": VprotonPerp / 1e3,
        "ValphaPara": ValphaPara / 1e3,
        "ValphaPerp": ValphaPerp / 1e3,
        "Vap": np.linalg.norm(Vap) / 1e3,
        "VA": VA,
        "TparaPcore": TparaProtonCore,
        "TperpPcore": TperpProtonCore,
        "TparaPbeam": TparaProtonBeam,
        "TperpPbeam": TperpProtonBeam,
        "TparaProton": TparaProton,
        "TperpProton": TperpProton,
        "TparaAlpha": TparaAlpha,
        "TperpAlpha": TperpAlpha,
        "Temperature Anisotropy": TperpProton / TparaProton,
        "Alpha Temperature Anisotropy": TperpAlpha / TparaAlpha,
        "Tap_ratio": Tap_ratio,
        "Theta_Vdrift_B": best_theta, 
    }

    # Save the moments, why not.
    with open(result_path+'/moments.txt', 'w') as f:
        for key, value in moments.items():
            f.write(f"{key}: {value}\n")

    # Save the results. Even fail, we save it.
    save_pickle(path=result_path+'/Protons.pkl', data=Protons_current)
    save_pickle(path=result_path+'/Alphas.pkl', data=Alphas_current)

    gc.collect()

    return Protons_current, Alphas_current


def main():
    # Specify the resolution of PAS during your interval
    # Usually 4.0 for early observations, and 2.0 for more recent observations.
    dt_seconds = 2.0

    # t start should be the the time of the initial separation + 4s.
    t_start = datetime(2023, 10, 2, 9, 8, 53)
    hhmmss = (t_start - timedelta(seconds=dt_seconds)).strftime("%H%M%S")
    t_end = datetime(2023, 10, 2, 10, 0, 0)
    yymmdd = t_start.strftime('%Y%m%d')
    data_list = os.listdir(f'data/SO/{yymmdd}')
    
    # Load the data, change your path here. Please correspond to the time you set.
    vdf_fname = next(file for file in data_list if 'pas-vdf' in file and not file.startswith('._'))
    grnd_fname = next(file for file in data_list if 'pas-grnd-mom' in file and not file.startswith('._'))
    mag_fname = next(file for file in data_list if 'mag-srf-normal' in file and not file.startswith('._'))
    #eflux_fname = next(file for file in data_list if 'pas-eflux' in file and not file.startswith('._'))
    count_fname = next(file for file in data_list if 'pas-3d' in file and not file.startswith('._'))

    vdf_cdffile = pycdf.CDF(f'data/SO/{yymmdd}/{vdf_fname}')
    grnd_cdffile = pycdf.CDF(f'data/SO/{yymmdd}/{grnd_fname}')
    mag_cdffile = pycdf.CDF(f'data/SO/{yymmdd}/{mag_fname}')
    #eflux_cdffile = pycdf.CDF(f'data/SO/{yymmdd}/{eflux_fname}')
    count_cdffile = pycdf.CDF(f'data/SO/{yymmdd}/{count_fname}')
    
    # Calculate the one-particle noise level.
    # one_particle_noise_level = OneParticleNoiseLevel(count_cdffile, vdf_cdffile)
    loaded_data = np.load(f'result/SO/{yymmdd}/one_particle_noise_level.npz')
    one_particle_noise_level = loaded_data['noise_level']

    # Initial Values
    Protons_initial = read_pickle(f'result/SO/{yymmdd}/Particles/Ions/{hhmmss}/Protons.pkl')
    Alphas_initial = read_pickle(f'result/SO/{yymmdd}/Particles/Ions/{hhmmss}/Alphas.pkl')
    
    # From the time range, we can find the indices of the data.
    epoch_vdf = vdf_cdffile['Epoch'][...]
    idx_times = FindIndexinInterval(t_start, t_end, epoch_vdf)

    for idx_time in idx_times:
        tslice = idx_time[1]
        yyyymmdd = tslice.strftime("%Y%m%d")
        hhmmss = tslice.strftime("%H%M%S")
        result_path = 'result/SO/' + yyyymmdd + '/Particles/Ions/' + hhmmss
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        try: 
            Protons_initial, Alphas_initial = all_process(idx_time, Protons_initial, Alphas_initial, vdf_cdffile, grnd_cdffile, mag_cdffile, one_particle_noise_level, result_path)
            # time.sleep()   # sleep for 2 seconds to avoid the laptop to overheat.
        except Exception as e:
            print(f'Error at {tslice}:, {e}')
            continue
    
    return 0

if __name__ == "__main__":
    main()