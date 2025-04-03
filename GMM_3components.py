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

def plot_one(ax1, ax2, vel, vdf_total, f_core, f_beam, f_alpha, co_type, scores):
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
    ax1.text(
        0.4, 0.9,
        'AIC: {:.2f}\nBIC: {:.2f}'.format(scores[0], scores[1]),
        fontsize=10,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax1.transAxes
    )
    ax1.set_ylim(-12, -6)
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
    ax2.set_ylim(-12, -6)
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
        "random_state": 0,
        "covariance_type": co_type,
        "means_init": initial_means,
        "weights_init": initial_weights,
    }

    gmm = GaussianMixture(**gmm_kwargs).fit(X)
    probas = gmm.predict_proba(X)


    f_all = [np.zeros_like(vdf_corrected) for _ in range(n_component)]

    for i in range(n_component):
        f_all[i][non_zero_idx] = probas[:, i] * vdf_corrected[non_zero_idx]
    
    component_index = [0, 1, 2]
    sort_indices = sorted(range(len(f_all)), key=lambda i: np.mean(f_all[i][non_zero_idx]))

    # Sort covariance using the same indices
    f_all_sorted = [f_all[i] for i in sort_indices]
    covariance_sorted = [gmm.covariances_[i] for i in sort_indices]
    means_sorted = [gmm.means_[i] for i in sort_indices]
    weights_sorted = [gmm.weights_[i] for i in sort_indices]

    # Set small values to 0 in f_all.
    for f in f_all_sorted:
        f[f < 1e-14] = 0

    # Get the BIC scores.
        k_dict = {
        'full': 62,     # 3 * 5 + 3 * (5 * (5 = 1)) / 2 + 2 
        'spherical': 20,    # 3 * 5 + 3 + 2 
        'diag': 32,    # 3 * 5 + 3 * 5 + 2
        'tied': 32,    # 3 * 5 + 3 + 2
    }

    n = len(non_zero_idx[0])
    AIC_score = - 2 * gmm.score(X) + 2 * k_dict[co_type]
    BIC_score = k_dict[co_type] * np.log(n) - 2 * gmm.score(X)
    scores = [AIC_score, BIC_score]

    return f_all_sorted, [means_sorted, covariance_sorted, weights_sorted], probas, scores


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
    ax.set_ylim(-12, -6)
    for idx, (xi, yi) in enumerate(zip(x.value, y)):
        ax.annotate(str(np.where(mask)[0][idx]), (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
    ax.set_xlabel('Vel [km/s]')
    ax.set_ylabel('log10(VDF)')
    plt.savefig(result_path + '/Corrected_VDF_1D.png')
    plt.close()

    def Remove_LowHigh_Speed_part(vdf_corrected, window_size=4, gap_limit=2):
        # Remove the low speed noise part.
        # vdf_corrected: vdf that is after one-particle noise level correction.
        # window size: how many succesive points of increasing can we consider as non_noise (low-speed), default 4.
        # gap_limit: how many points of gap can we consider as non_noise (low-speed), default 4.

        data = np.sum(vdf_corrected, axis=(0, 1))
        y = log10_vdf(data)
        lower_speed_indice = len(data) - 1

        for i in range(len(data) - 1, window_size - 2, -1):  # Start from the end and move backward
            if all(data[j] < data[j - 1] for j in range(i, i - window_size + 1, -1)):  # Check for increases
                lower_speed_indice = i
                break
        vdf_corrected[:, :, lower_speed_indice:] = 0

        #IncreasingPoints = np.arange(lower_speed_indice - 1, lower_speed_indice - (window_size + 1), -1)[::-1]
        #Gaps = [np.abs(y[IncreasingPoints[i+1]] - y[IncreasingPoints[i]]) for i in range(len(IncreasingPoints)-1)][::-1]

        #for gap in Gaps:
        #   if gap > 0.5:
        #        lower_speed_indice -= 1

        data = np.sum(vdf_corrected, axis=(0, 1))
        mask = y != 0
        true_indexes = np.where(mask)[0][:10]

        higher_speed_indice = true_indexes[0]

        for i in range(len(true_indexes) - 1):
            # Check the gap bwtween successive points.
            if true_indexes[i+1] - true_indexes[i] > gap_limit:
                higher_speed_indice = true_indexes[i+1]
        
        vdf_corrected[:, :, :higher_speed_indice] = 0
        
        return vdf_corrected

    vdf_corrected = Remove_LowHigh_Speed_part(vdf_corrected)

    # Plot to see the 1D vdf.
    y = log10_vdf(np.sum(vdf_corrected, axis=(0, 1)))
    mask = y != 0
    x = vel[mask]
    y = y[mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, label='Corrected VDF')
    ax.scatter(x, y, s=20, color='red')
    ax.set_ylim(-12, -6)
    for idx, (xi, yi) in enumerate(zip(x.value, y)):
        ax.annotate(str(np.where(mask)[0][idx]), (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
    ax.set_xlabel('Vel [km/s]')
    ax.set_ylabel('log10(VDF)')
    plt.savefig(result_path + '/Corrected_VDF_Cleaned.png')
    plt.close()

    # Find the dividing index:
    vdf_corrected_1D = np.sum(vdf_corrected, axis=(0, 1))
    mask = vdf_corrected_1D > 0
    x = vel[mask]
    y = np.log10(vdf_corrected_1D[mask])

    sigma = 1.0
    smoothed_y = gaussian_filter1d(y, sigma)
    local_minima_indices = argrelextrema(smoothed_y, np.less)[0]

    extreme_min_index = None
    if local_minima_indices.size > 0:
        extreme_min_index = local_minima_indices[-1]

    if extreme_min_index is None:
        dividing_idx = int((np.where(mask)[0][0] + np.where(mask)[0][-1]) / 2.0)
        print("Failed to find the dividing index, use the middle point instead.")
        print("Dividing idx: ", dividing_idx)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y, label='Corrected VDF')
        ax.plot(x, smoothed_y, label='Smoothed VDF', color='green')
        ax.scatter(x, y, s=20, color='red')
        ax.axvline(x[extreme_min_index].value, color='black', linestyle='--', label='Dividing Index')
        ax.legend()
        ax.set_xlabel('Vel [km/s]')
        ax.set_ylabel('log10(VDF)')
        for idx, (xi, yi) in enumerate(zip(x.value, y)):
            ax.annotate(str(np.where(mask)[0][idx]), (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
        plt.savefig(result_path + '/Corrected_VDF_1D_dividing.png')
        plt.close()

        # Find the dividing index:
        vels = vel.value
        x_val = x[extreme_min_index].value
        dividing_idx = len(vels)
        if extreme_min_index is not None:
            dividing_idx = np.where(vels == x_val)[0][0]
        print('Dividing index: ', dividing_idx)

    # Get the initial values for GMM.
    u_proton_core = cal_bulk_velocity_Spherical(Protons_initial, component='core')
    u_proton_core_Baligned = rotateVectorIntoFieldAligned(u_proton_core[0], u_proton_core[1], u_proton_core[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    u_proton_beam = cal_bulk_velocity_Spherical(Protons_initial, component='beam')
    u_proton_beam_Baligned = rotateVectorIntoFieldAligned(u_proton_beam[0], u_proton_beam[1], u_proton_beam[2], Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    u_alpha = cal_bulk_velocity_Spherical(Alphas_initial)
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
    f_full, dist_paras_full, probas_full, scores_full = cal_GMM(V_para, V_perp1, V_perp2, vdf_corrected, 'full', initial_means, n_component)
    f_alpha_full, f_beam_full, f_core_full = f_full
    f_diag, dist_paras_diag, probas_diag, scores_diag = cal_GMM(V_para, V_perp1, V_perp2, vdf_corrected, 'diag', initial_means, n_component)
    f_alpha_diag, f_beam_diag, f_core_diag = f_diag
    f_spherical, dist_paras_spherical, probas_spherical, scores_spherical = cal_GMM(V_para, V_perp1, V_perp2, vdf_corrected, 'spherical', initial_means, n_component)
    f_alpha_spherical, f_beam_spherical, f_core_spherical = f_spherical
    f_tied, dist_paras_tied, probas_tied, scores_tied = cal_GMM(V_para, V_perp1, V_perp2, vdf_corrected, 'tied', initial_means, n_component)
    f_alpha_tied, f_beam_tied, f_core_tied = f_tied

    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    # Full 
    plot_one(ax[0, 0], ax[1, 0], vel, vdf_corrected, f_core_full, f_beam_full, f_alpha_full, 'Full', scores=scores_full)
    # diag
    plot_one(ax[0, 1], ax[1, 1], vel, vdf_corrected, f_core_diag, f_beam_diag, f_alpha_diag, 'Diag', scores=scores_diag)
    # tied
    plot_one(ax[0, 2], ax[1, 2], vel, vdf_corrected, f_core_tied, f_beam_tied, f_alpha_tied, 'Tied', scores=scores_tied)
    # spherical
    plot_one(ax[0, 3], ax[1, 3], vel, vdf_corrected, f_core_spherical, f_beam_spherical, f_alpha_spherical, 'Spherical', scores=scores_spherical)
    plt.savefig(result_path + '/GMM_all.png')
    plt.close() 

    def remove_alpha_low_speed(f_alpha, co_type):
        data = np.sum(f_alpha, axis=(0, 1))
        mask = data > 0
        x = vel[mask]
        y = np.log10(data[mask])
        
        # find the most extreme local minima, that's the dividing index.
        sigma = 1.5
        smoothed_y = gaussian_filter1d(y, sigma)
        local_minima_indices = argrelextrema(smoothed_y, np.less)[0]
        extreme_min_index = None
        if local_minima_indices.size > 0:
            extreme_min_index = local_minima_indices[np.argmin(y[local_minima_indices])]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y, label='Alpha', color='green')
        ax.plot(x, smoothed_y, label='Smoothed Alpha', color='blue')
        ax.scatter(x, y, s=20, color='green')
        if extreme_min_index is not None:
            ax.scatter(x[extreme_min_index], y[extreme_min_index], s=40, color='red')
        ax.legend()
        ax.set_title(co_type)
        ax.set_xlabel('Vel [km/s]')
        ax.set_ylabel('log10(VDF)')
        plt.savefig(result_path + '/Alpha_' + co_type + '_1D.png')
        plt.close()

        # Find the dividing index:
        vels = vel.value
        x_val = x[extreme_min_index].value
        alpha_dividing_idx = len(vels)
        if extreme_min_index is not None:
            alpha_dividing_idx = np.where(vels == x_val)[0][0]

        return alpha_dividing_idx

    alpha_dividing_idx = 96
    while alpha_dividing_idx > dividing_idx:
        print('above ' + str(alpha_dividing_idx) + ' set to 0') 
        f_alpha_diag[:, :, alpha_dividing_idx:] = 0
        idx_new = remove_alpha_low_speed(f_alpha_diag, 'Diag')
        if idx_new == alpha_dividing_idx:
            break
        print(idx_new)
        alpha_dividing_idx = idx_new

    beam_dividing_idx = 96
    while beam_dividing_idx > dividing_idx:
        print('above ' + str(beam_dividing_idx) + ' set to 0') 
        f_beam_diag[:, :, beam_dividing_idx:] = 0
        idx_new = remove_alpha_low_speed(f_beam_diag, 'Diag Beam')
        if idx_new == beam_dividing_idx:
            break
        print(idx_new)
        beam_dividing_idx = idx_new

    alpha_dividing_idx = 96
    while alpha_dividing_idx > dividing_idx:
        print('above ' + str(alpha_dividing_idx) + ' set to 0') 
        f_alpha_spherical[:, :, alpha_dividing_idx:] = 0
        idx_new = remove_alpha_low_speed(f_alpha_spherical, 'Spherical')
        if idx_new == alpha_dividing_idx:
            break
        print(idx_new)
        alpha_dividing_idx = idx_new

    beam_dividing_idx = 96
    while beam_dividing_idx > dividing_idx:
        print('above ' + str(beam_dividing_idx) + ' set to 0') 
        f_beam_spherical[:, :, beam_dividing_idx:] = 0
        idx_new = remove_alpha_low_speed(f_beam_spherical, 'Spherical Beam')
        if idx_new == beam_dividing_idx:
            break
        print(idx_new)
        beam_dividing_idx = idx_new

    # Carry on with diag and spherical.
    # Let's carry on with diag and spherical.
    Protons_diag = SolarWindParticle('proton', time=tslice_vdf, magfield=magF_SRF, grid=[theta.value, phi.value, vel.value * 1e3], coord_type='Spherical')
    Protons_diag.set_vdf(f_core_diag, component='core')
    Protons_diag.set_vdf(f_beam_diag, component='beam')
    Alphas_diag = SolarWindParticle('alpha', time=tslice_vdf, magfield=magF_SRF, grid=[theta.value, phi.value, vel.value * 1e3], coord_type='Spherical')
    Alphas_diag.set_vdf(f_alpha_diag)

    Protons_spherical = SolarWindParticle('proton', time=tslice_vdf, magfield=magF_SRF, grid=[theta.value, phi.value, vel.value * 1e3], coord_type='Spherical')
    Protons_spherical.set_vdf(f_core_spherical, component='core')
    Protons_spherical.set_vdf(f_beam_spherical, component='beam')
    Alphas_spherical = SolarWindParticle('alpha', time=tslice_vdf, magfield=magF_SRF, grid=[theta.value, phi.value, vel.value * 1e3], coord_type='Spherical')
    Alphas_spherical.set_vdf(f_alpha_spherical)

    # Print the moments to see what the results look like.
    Vpcore_diag = cal_bulk_velocity_Spherical(Protons_diag, 'core') / 1e3
    Vpbeam_diag = cal_bulk_velocity_Spherical(Protons_diag, 'beam') / 1e3
    Valpha_diag = cal_bulk_velocity_Spherical(Alphas_diag) / (np.sqrt(2) * 1e3)
    Vpcore_spherical = cal_bulk_velocity_Spherical(Protons_spherical, 'core') / 1e3
    Vpbeam_spherical = cal_bulk_velocity_Spherical(Protons_spherical, 'beam') / 1e3
    Valpha_spherical = cal_bulk_velocity_Spherical(Alphas_spherical) / (np.sqrt(2) * 1e3)

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

    # Calculate the angle between drif speeds with respect to the magnetic field.
    Theta_PCore_Beam_Diag = np.arctan((VpbeamDiagPerp - VpcoreDiagPerp) / (VpbeamDiagPara - VpcoreDiagPara)) * 180 / np.pi
    Theta_PCore_Alpha_Diag = np.arctan((ValphaDiagPerp - VpcoreDiagPerp) / (ValphaDiagPara - VpcoreDiagPara)) * 180 / np.pi

    Theta_PCore_Beam_Spherical = np.arctan((VpbeamSphericalPerp - VpcoreSphericalPerp) / (VpbeamSphericalPara - VpcoreSphericalPara)) * 180 / np.pi
    Theta_PCore_Alpha_Spherical = np.arctan((ValphaSphericalPerp - VpcoreSphericalPerp) / (ValphaSphericalPara - VpcoreSphericalPara)) * 180 / np.pi

    print('Diag')
    print('=================================')
    print('Vpcore:', VpcoreDiagPara, VpcoreDiagPerp)
    print('Vpbeam:', VpbeamDiagPara, VpbeamDiagPerp)
    print('Vpalpha:', ValphaDiagPara, ValphaDiagPerp)

    print('Spherical')
    print("=================================")
    print('Vpcore:', VpcoreSphericalPara, VpcoreSphericalPerp)
    print('Vpbeam:', VpbeamSphericalPara, VpbeamSphericalPerp)
    print('Vpalpha:', ValphaSphericalPara, ValphaSphericalPerp)

    print("Beam-Core Theta Diag: ", Theta_PCore_Beam_Diag)
    print("Alpha-Core Theta Diag: ", Theta_PCore_Alpha_Diag)
    print("Beam-Core Theta Spherical: ", Theta_PCore_Beam_Spherical)
    print("Alpha-Core Theta Spherical: ", Theta_PCore_Alpha_Spherical)

    # Compute the norms of Vpcore and Valpha
    norm_Vpcore_diag = np.linalg.norm(Vpcore_diag)
    norm_Valpha_diag = np.linalg.norm(Valpha_diag)
    norm_Vpcore_spherical = np.linalg.norm(Vpcore_spherical)
    norm_Valpha_spherical = np.linalg.norm(Valpha_spherical)

    # First criterion: Ensure norm(Vpcore) < norm(Valpha)
    #if norm_Vpcore_diag > norm_Valpha_diag and norm_Vpcore_spherical > norm_Valpha_spherical:
    #    print("Error: Both Diag and Spherical have Vpcore > Valpha. Use the previous one.")
    #    Protons_current = Protons_initial
    #    Alphas_current = Alphas_initial

    #elif norm_Vpcore_diag < norm_Valpha_diag and norm_Vpcore_spherical > norm_Valpha_spherical:
    #    print("Diag satisfies Vpcore < Valpha, selecting Diag.")
    #    Protons_current = Protons_diag
    #    Alphas_current = Alphas_diag
    #    which_one = 'diag'
    #    Scores_current = scores_diag

    #elif norm_Vpcore_diag > norm_Valpha_diag and norm_Vpcore_spherical < norm_Valpha_spherical:
    #    print("Spherical satisfies Vpcore < Valpha, selecting Spherical.")
    #    Protons_current = Protons_spherical
    #    Alphas_current = Alphas_spherical
    #    which_one = 'spherical'
    #    Scores_current = scores_spherical
    #else:
        # Both satisfy the condition, use Theta_PCore_Alpha comparison

    # Which one is more aligned with B is selected.
    if np.abs(Theta_PCore_Alpha_Diag) < np.abs(Theta_PCore_Alpha_Spherical):
        print("Diag is better.")
        which_one = 'diag'
        Protons_current = Protons_diag
        Alphas_current = Alphas_diag
        Scores_current = scores_diag
    else:
        print("Spherical is better.")
        which_one = 'spherical'
        Protons_current = Protons_spherical
        Alphas_current = Alphas_spherical
        Scores_current = scores_spherical


    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plot_one(ax[0], ax[1], vel, vdf_corrected, Protons_current.get_vdf(component='core'), Protons_current.get_vdf(component='beam'), Alphas_current.get_vdf(), 'Final', scores=Scores_current)
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
    Valpha = cal_bulk_velocity_Spherical(Alphas_current) / np.sqrt(2)
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

    # Theta
    Theta_CoreBeam = np.arctan((VpbeamPerp - VpcorePerp) / (VpbeamPara - VpcorePara)) * 180 / np.pi
    Theta_CoreAlpha = np.arctan((ValphaPerp - VpcorePerp) / (ValphaPara - VpcorePara)) * 180 / np.pi

    # Save the important parameter printed to a txt file.
    moments = {
        "Which one": which_one,
        'mag field SRF': magF_SRF,
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
        "TparaPcore": TparaProton,
        "TperpPcore": TperpProton,
        "TparaAlpha": TparaAlpha,
        "TperpAlpha": TperpAlpha,
        "Temperature Anisotropy": TperpProton / TparaProton,
        "Alpha Temperature Anisotropy": TperpAlpha / TparaAlpha,
        "Tap_ratio": Tap_ratio,
        "Theta_BeamCore_B": Theta_CoreBeam,
        "Theta_AlphaCore_B": Theta_CoreAlpha, 
        "AIC $ BIC diag": scores_diag,
        "AIC $ BIC spherical": scores_spherical
    }

    # Save the moments, why not.
    with open(result_path+'/moments.txt', 'w') as f:
        for key, value in moments.items():
            f.write(f"{key}: {value}\n")

    # Save the results. Even fail, we save it.
    save_pickle(path=result_path+'/Protons.pkl', data=Protons_current)
    save_pickle(path=result_path+'/Alphas.pkl', data=Alphas_current)

    # If the separation fails, in order to avoid the failure of the next separation, we use the previous separation as the initial value and return it to the next separation.

    if np.abs(Theta_CoreAlpha) > 30:
        print("Separation failed, use the previous separation.")
        Protons_current = Protons_initial
        Alphas_current = Alphas_initial

    return Protons_current, Alphas_current

def main():
    # 110258 went wrong at the beginning.
    t_start = datetime(2023, 3, 19, 14, 0, 2)
    hhmmss = (t_start - timedelta(seconds=4)).strftime("%H%M%S")
    t_end = datetime(2023, 3, 19, 15, 0, 0)
    yymmdd = t_start.strftime('%Y%m%d')
    data_list = os.listdir(f'data/SO/{yymmdd}')

    # Load the data, change your path here. Please correspond to the time you set.
    vdf_fname = next(file for file in data_list if 'pas-vdf' in file and not file.startswith('._'))
    grnd_fname = next(file for file in data_list if 'pas-grnd-mom' in file and not file.startswith('._'))
    mag_fname = next(file for file in data_list if 'mag-srf-normal' in file and not file.startswith('._'))
    eflux_fname = next(file for file in data_list if 'pas-eflux' in file and not file.startswith('._'))
    count_fname = next(file for file in data_list if 'pas-3d' in file and not file.startswith('._'))

    vdf_cdffile = pycdf.CDF(f'data/SO/{yymmdd}/{vdf_fname}')
    grnd_cdffile = pycdf.CDF(f'data/SO/{yymmdd}/{grnd_fname}')
    mag_cdffile = pycdf.CDF(f'data/SO/{yymmdd}/{mag_fname}')
    eflux_cdffile = pycdf.CDF(f'data/SO/{yymmdd}/{eflux_fname}')
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
            print('Error at {tslice}:, {e}')
            continue
    
    return 0

if __name__ == "__main__":
    main()


