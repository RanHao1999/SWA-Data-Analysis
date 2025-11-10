"""
In this code, we generate the input file (the f0-tables) that ALPS needs to run the simulation.
This is based on the user has already read the data and saved distribution of different particles as SolarWindParticle object.

Method:
We get all the measured points from PAS, and interpolate them and merge the interpolation into a bi-Maxwellian distribution background.
The background bi-Maxwellian distribution is called a "collar".
Every step is carefully recorded, and the code is well documented.

author = @RanHao1999, at UCL Space & Climate Physics, Mullard Space Science Laboratory
email = ranhaogm@gmail.com; hao.ran.24@ucl.ac.uk
"""

# import the necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
import sys

from Funcs import *
from SolarWindPack import *
from scipy.interpolate import griddata
import shutil
import subprocess
import matplotlib.colors as colors
import pickle

from scipy.spatial import ConvexHull
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter

# To work properly with the relative path
os.chdir(sys.path[0])

# functions
def average_SW_particle(SW_particles):
    """
    SW_particles is a list containing the particles that the user wants to average.
    Please make sure that you have removed the noise before averaging.
    """
    vel = SW_particles[0].grid['velocity']
    theta = SW_particles[0].grid['elevation']
    phi = SW_particles[0].grid['azimuth']
    species = SW_particles[0].species
    coord_type = SW_particles[0].coord_type

    time_lst = []
    for particle in SW_particles:
        time_lst.append(particle.time)
    
    # Average Mag fields.
    B_fields_list = []
    for particle in SW_particles:
        B_fields_list.append(particle.magfield)
    average_Bfield = np.mean(B_fields_list, axis=0)

    # Average VDF。
    # If species == "proton", then average beam and core separately.
    # If species == "alpha", then average the whole VDF.
    if species == 'proton':
        Core_VDFs = []
        Beam_VDFs = []
        for particle in SW_particles:
            core_vdf = particle.get_vdf(component='core')
            beam_vdf = particle.get_vdf(component='beam')
            Core_VDFs.append(core_vdf)
            Beam_VDFs.append(beam_vdf)
        average_Core_VDF = np.mean(Core_VDFs, axis=0)
        average_Beam_VDF = np.mean(Beam_VDFs, axis=0)
        Particle_out = SolarWindParticle(species, time=time_lst, grid=[theta, phi, vel], magfield=average_Bfield, coord_type=coord_type)
        Particle_out.set_vdf(average_Core_VDF, component='core')
        Particle_out.set_vdf(average_Beam_VDF, component='beam')
    elif species == 'alpha':
        VDFs = []
        for particle in SW_particles:
            vdf = particle.get_vdf()
            VDFs.append(vdf)
        average_VDF = np.mean(VDFs, axis=0)
        Particle_out = SolarWindParticle(species, time=time_lst, grid=[theta, phi, vel], magfield=average_Bfield, coord_type=coord_type)
        Particle_out.set_vdf(average_VDF)
    elif species == 'electron':
        VDFs = []
        for particle in SW_particles:
            vdf = particle.get_vdf()
            VDFs.append(vdf)
        average_VDF = np.mean(VDFs, axis=0)
        Particle_out = SolarWindParticle(species, time=time_lst, grid=[theta, phi, vel], magfield=average_Bfield, coord_type=coord_type)
        Particle_out.set_vdf(average_VDF)
    else:
        raise ValueError("Species not recognized.")
    
    return Particle_out

def bi_maxwellian(v_par, v_perp, n, T_par, T_perp, m, drift_par=0.0):
    """
    Evaluate bi-Maxwellian distribution at a given (v_par, v_perp)

    Parameters:
    - v_par: float or array, parallel velocity (m/s)
    - v_perp: float or array, perpendicular velocity (m/s)
    - n: number density (m^-3)
    - T_par: parallel temperature (eV)
    - T_perp: perpendicular temperature (eV)
    - m: particle mass (kg)
    - drift_par: parallel drift velocity (m/s), optional

    Returns:
    - f: bi-Maxwellian value
    """
    kB = 1.380649e-23  # J/K
    eV_to_K = 11604.525  # K per eV
    T_par_K = T_par * eV_to_K
    T_perp_K = T_perp * eV_to_K

    v_th_par = np.sqrt(2 * kB * T_par_K / m)
    v_th_perp = np.sqrt(2 * kB * T_perp_K / m)

    coeff = n / (np.pi**1.5 * v_th_perp**2 * v_th_par)
    exponent = -((v_par - drift_par)**2 / v_th_par**2 + v_perp**2 / v_th_perp**2)
    return coeff * np.exp(exponent)

def renormalize_vdf(vdf, vpar_grid, vper_grid, target_density):
    # After interpolation, we need to renormalize the VDF to match the target (original) density.
    # This will slightlt change the VDF values.
    delta_v_para = vpar_grid[0, 1] - vpar_grid[0, 0]
    delta_v_perp = vper_grid[1, 0] - vper_grid[0, 0]

    current_density = 2 * np.pi * np.sum(vdf * vper_grid) * delta_v_para * delta_v_perp
    normalization_factor = target_density / current_density
    normalized_vdf = vdf * normalization_factor

    return normalized_vdf

def generate_mask_from_points(vpar_data, vperp_data, vpar_grid, vperp_grid):
    """
    Generate a boolean mask that includes the convex hull of the measured points
    and enforces closure at V_perp = 0 (x-axis), filling in the region below due to instrumental cutoff.
    """
    # Original measured points
    points = np.vstack((vpar_data, vperp_data)).T

    # Find min and max V_parallel among measured points
    vpar_min = np.min(vpar_data)
    vpar_max = np.max(vpar_data)

    # Add two anchor points on the bottom boundary
    bottom_points = np.array([
        [vpar_min, 0],  # left base
        [vpar_max, 0],  # right base
    ])

    # Combine original + artificial base points
    extended_points = np.vstack([points, bottom_points])

    # Convex hull of extended set
    hull = ConvexHull(extended_points)
    hull_path = Path(extended_points[hull.vertices])

    # Grid points to be checked
    flat_grid_points = np.vstack((vpar_grid.ravel(), vperp_grid.ravel())).T
    inside = hull_path.contains_points(flat_grid_points)

    # Reshape to original grid shape
    mask = inside.reshape(vpar_grid.shape)

    return mask



def smooth_mask(mask, sigma=3):
    """
    Smooths the binary mask to create a smooth transition between
    measured and model regions.
    
    Parameters:
        mask (2D array): Boolean mask from convex hull (True=measured region)
        sigma (float): Gaussian kernel width for smoothing
        
    Returns:
        smooth_weight (2D array): Values between 0 and 1 for blending
    """
    # Convert mask to float
    mask_float = mask.astype(float)
    
    # Apply Gaussian filter
    smoothed = gaussian_filter(mask_float, sigma=sigma)
    
    # Normalize so values are between 0 and 1
    smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    
    return smoothed

def merge_vdf_logspace(vdf_interp, biMax, smooth_weight):
    """
    Merge interpolated VDF with biMaxwellian in log-space, safely.
    Uses log10(f) blending only where interpolation is valid.
    
    Parameters:
        vdf_interp (2D array): interpolated VDF (may have NaNs or 0s)
        biMax (2D array): analytic background
        smooth_weight (2D array): 0 to 1 blending mask
    
    Returns:
        vdf_final (2D array): combined VDF
    """
    # Default to biMax everywhere
    vdf_final = biMax.copy()

    # Valid blending zone: interp is finite and smooth_weight > 0.01
    valid_mask = np.isfinite(vdf_interp) & (smooth_weight > 0.01)

    # Clip values to avoid log(0)
    f_interp_clipped = np.clip(vdf_interp[valid_mask], 1e-30, None)
    f_biMax_clipped = np.clip(biMax[valid_mask], 1e-30, None)

    # Blend in log space
    log_interp = np.log10(f_interp_clipped)
    log_biMax = np.log10(f_biMax_clipped)
    weight = smooth_weight[valid_mask]

    log_blend = weight * log_interp + (1 - weight) * log_biMax
    vdf_final[valid_mask] = 10 ** log_blend

    return vdf_final


def main():
    # Some constants.
    # mass of particles.
    mp = 1.67262192e-27 # kg
    ma = 6.64216e-27    # kg
    me = 9.10938356e-31 # kg
    e = 1.602176634e-19  # elementary charge in Coulombs
    # Grid for interpolation
    n_perp = 120
    n_para = 240

    # Specify the time range, this code will average the measurements in this time range, and feed into ALPS.
    # Recommend: 1 min gap.
    tstart = datetime(2022, 10, 23, 2, 6, 0)
    tend = datetime(2022, 10, 23, 2, 7, 0)
    smooth_sigma = 10
    yymmdd = tstart.strftime("%Y%m%d")
    hhmmss_str = tstart.strftime("%H%M%S") + "To" + tend.strftime("%H%M%S")

    # This is the directory for saving results.
    res_dir = f'result/SO/{yymmdd}/4ALPS/{hhmmss_str}/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    # Helper function to check if a string is a valid time format
    def is_valid_time_string(time_str):
        if time_str.startswith("._"):   # stupid MacOS file system
            return False
        try:
            time_obj = datetime.strptime(time_str, "%H%M%S").time()
            return tstart.time() <= time_obj <= tend.time()
        except ValueError:
            return False
    
    ion_hhmmss_list = [time_str for time_str in os.listdir(f'result/SO/{yymmdd}/Particles/Ions/') if is_valid_time_string(time_str) and os.path.exists(f'result/SO/{yymmdd}/Particles/Ions/{time_str}/Protons.pkl')]

    # Read ions and save them into a list.
    Proton_list = []
    Alpha_list = []
    for hhmmss in ion_hhmmss_list:
        Proton = read_pickle(f'result/SO/{yymmdd}/Particles/Ions/{hhmmss}/Protons.pkl')
        Alpha = read_pickle(f'result/SO/{yymmdd}/Particles/Ions/{hhmmss}/Alphas.pkl')
        Proton_list.append(Proton)
        Alpha_list.append(Alpha)
    # Average to get the final particle distribution.
    Protons = average_SW_particle(Proton_list)
    Alphas = average_SW_particle(Alpha_list)

    # Save them into our result_directory.
    save_pickle(path=res_dir + 'Protons.pkl', data=Protons)
    save_pickle(path=res_dir + 'Alphas.pkl', data=Alphas)

    # Since some constants need to be calculated from the ion distribution, let's handle the ions first.
    # For ions, since we have separated the different components. 
    # We need the one-particle-noise level to be removed from the obtained results.
    loaded_data = np.load(f'result/SO/{yymmdd}/one_particle_noise_level.npz')
    one_particle_noise_level = loaded_data['noise_level']

    # Remove the one-particle-noise level and replace the original data.
    pcore_vdf = Protons.get_vdf(component='core')
    pbeam_vdf = Protons.get_vdf(component='beam')
    alpha_vdf = Alphas.get_vdf()

    pcore_vdf[pcore_vdf < one_particle_noise_level] = 0
    pbeam_vdf[pbeam_vdf < one_particle_noise_level] = 0
    alpha_vdf[alpha_vdf < one_particle_noise_level] = 0

    Protons.set_vdf(pcore_vdf, component='core')
    Protons.set_vdf(pbeam_vdf, component='beam')
    Alphas.set_vdf(alpha_vdf)

    # Transfer the Cartesian VDF to FieldAligned VDF.
    TMatrix = np.array([[-1, 0, 0], 
                        [0, 1, 0], 
                        [0, 0, -1]])
    # Bulk velocity of protons. 
    # Particles will be shifted to the proton-rest frame.
    V_bulk_SRF = cal_bulk_velocity_Spherical(Protons)

    # Protons.
    Protons_FieldAligned = transferToFieldAligned(Protons, transfer_Matrix=TMatrix, VPbulk_SRF=V_bulk_SRF)

    # Alphas.
    # A bit more complicated due to the sqrt(2) shift in the spectrum
    Alphas_FieldAligned = transferToFieldAligned(Alphas, transfer_Matrix=TMatrix, VPbulk_SRF=V_bulk_SRF)

    # Critical parameters that we need:
    # Density
    N_proton = cal_density_Spherical(Protons)
    N_alpha = cal_density_Spherical(Alphas)
    N_electron = N_proton + 2 * N_alpha     # quasi-neutrality.

    # Magfield
    B_srf_0 = Protons.magfield[0]
    B_srf_1 = Protons.magfield[1]
    B_srf_2 = Protons.magfield[2]
    B_magnitude = np.sqrt(B_srf_0 ** 2 + B_srf_1 ** 2 + B_srf_2 ** 2)

    # Temperatures
    Tp_para, Tp_perp = Temperature_para_perp(Protons)
    Ta_para, Ta_perp = Temperature_para_perp(Alphas)
    Taniso_proton = Tp_perp / Tp_para
    Taniso_alpha = Ta_perp / Ta_para
    Tp = (Tp_para + 2 * Tp_perp) / 3
    Ta = (Ta_para + 2 * Ta_perp) / 3

    # Velocity
    Vp_srf_0 = cal_bulk_velocity_Spherical(Protons)[0]
    Vp_srf_1 = cal_bulk_velocity_Spherical(Protons)[1]
    Vp_srf_2 = cal_bulk_velocity_Spherical(Protons)[2]
    Va_srf_0 = cal_bulk_velocity_Spherical(Alphas)[0]
    Va_srf_1 = cal_bulk_velocity_Spherical(Alphas)[1]
    Va_srf_2 = cal_bulk_velocity_Spherical(Alphas)[2]

    (Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz) = fieldAlignedCoordinates(B_srf_0, B_srf_1, B_srf_2)
    (Vp_para, Vp_perp1, Vp_perp2) = rotateVectorIntoFieldAligned(Vp_srf_0, Vp_srf_1, Vp_srf_2, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    Vp_perp = np.sqrt(Vp_perp1 ** 2 + Vp_perp2 ** 2)
    (Va_para, Va_perp1, Va_perp2) = rotateVectorIntoFieldAligned(Va_srf_0, Va_srf_1, Va_srf_2, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    Va_perp = np.sqrt(Va_perp1 ** 2 + Va_perp2 ** 2)
    ua_drift = Va_para - Vp_para

    # Alfven Speed.
    mu_0 = 4 * np.pi * 1e-7
    V_Alfven = (B_magnitude * 1e-9) / np.sqrt(mu_0 * (N_proton * mp + N_alpha * ma))

    # plasma beta.
    # We need this because pho_p / dp = sqrt(beta_p / 2)
    kB = 1.380649e-23 # J/K

    beta_proton_para = (N_proton * kB * (Tp_para * 11604.5)) / ((B_magnitude * 1e-9) ** 2 / (2 * mu_0))
    beta_alpha_para = (N_alpha * kB * (Ta_para * 11604.5)) / ((B_magnitude * 1e-9) ** 2 / (2 * mu_0))

    beta_proton_perp = (N_proton * kB * (Tp_perp * 11604.5)) / ((B_magnitude * 1e-9) ** 2 / (2 * mu_0))
    beta_alpha_perp = (N_alpha * kB * (Ta_perp * 11604.5)) / ((B_magnitude * 1e-9) ** 2 / (2 * mu_0))

    beta_proton = (N_proton * kB * (Tp * 11604.5)) / ((B_magnitude * 1e-9) ** 2 / (2 * mu_0))
    beta_alpha = (N_alpha * kB * (Ta * 11604.5)) / ((B_magnitude * 1e-9) ** 2 / (2 * mu_0))

    # proton inertial length and Larmor radius.
    c = 2.998e8 # m/s, speed of light
    epsilon_0 = 8.854e-12 # F/m
    proton_charge = 1.6022e-19 # C
    dp = c / np.sqrt(N_proton * proton_charge**2 / (mp * epsilon_0))    # proton inertial length

    # Larmor radius.
    Vpth_perp = np.sqrt(2 * kB * Tp_perp * 11604.5 / mp) # m/s
    rho_p = (mp * Vpth_perp) / (proton_charge * (B_magnitude * 1e-9)) # m

    # Save as a directory and save into as csv file.
    parameters = {
        'N_proton': N_proton,
        'N_alpha': N_alpha,
        'N_electron': N_electron,
        'B_srf_0': B_srf_0,
        'B_srf_1': B_srf_1,
        'B_srf_2': B_srf_2,
        'B_magnitude': B_magnitude,
        'Tp_para': Tp_para,
        'Tp_perp': Tp_perp,
        'Ta_para': Ta_para,
        'Ta_perp': Ta_perp,
        'Taniso_proton': Taniso_proton,
        'Taniso_alpha': Taniso_alpha,
        'Tproton': Tp,
        'Talpha': Ta,
        'Vp_srf_0': Vp_srf_0,
        'Vp_srf_1': Vp_srf_1,
        'Vp_srf_2': Vp_srf_2,
        'Vp_para': Vp_para,
        'Vp_perp': Vp_perp, 
        'Va_srf_0': Va_srf_0,
        'Va_srf_1': Va_srf_1,
        'Va_srf_2': Va_srf_2,
        'Va_para': Va_para,
        'Va_perp': Va_perp,
        'V_Alfven': V_Alfven,
        'beta_proton_para': beta_proton_para,
        'beta_alpha_para': beta_alpha_para,
        'beta_proton_perp': beta_proton_perp,
        'beta_alpha_perp': beta_alpha_perp,
        'beta_proton': beta_proton,
        'beta_alpha': beta_alpha,
        'NGrid_perp': n_perp,
        'NGrid_para': n_para,
        'proton inertial length': dp,
        'proton Larmor radius': rho_p
    }

    # Before we interpolate, see what the measurements are actually like.
    # Ion measurements are like:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    vpara_p = Protons_FieldAligned.grid['Vpara']
    vperp_p = np.sqrt(Protons_FieldAligned.grid['Vperp1'] ** 2 + Protons_FieldAligned.grid['Vperp2'] ** 2)
    vdf_p = Protons_FieldAligned.get_vdf()
    mask_p = vdf_p > 0
    vpara_p = vpara_p[mask_p]
    vperp_p = vperp_p[mask_p]
    vdf_p = vdf_p[mask_p]

    cs1 = ax1.scatter(vpara_p / V_Alfven, vperp_p / V_Alfven, c=np.log10(vdf_p), cmap='viridis', s=10)
    ax1.set_xlabel('$V_{para}$ ')
    ax1.set_ylabel('$V_{perp}$')
    ax1.set_title('Protons')
    cbar1 = fig.colorbar(cs1, ax=ax1, label='log$_{10}$(VDF)')

    vpara_a = Alphas_FieldAligned.grid['Vpara']
    vperp_a = np.sqrt(Alphas_FieldAligned.grid['Vperp1'] ** 2 + Alphas_FieldAligned.grid['Vperp2'] ** 2)
    vdf_a = Alphas_FieldAligned.get_vdf()
    mask_a = vdf_a > 0
    vpara_a = vpara_a[mask_a]
    vperp_a = vperp_a[mask_a]
    vdf_a = vdf_a[mask_a]

    cs2 = ax2.scatter(vpara_a / V_Alfven, vperp_a / V_Alfven, c=np.log10(vdf_a), cmap='viridis', s=10)
    ax2.set_xlabel('$V_{para}$ / $V_A$')
    ax2.set_ylabel('$V_{perp}$ / $V_A$')
    ax2.set_title('Alphas')
    cbar2 = fig.colorbar(cs2, ax=ax2, label='log$_{10}$(VDF)')
    plt.savefig(res_dir + 'Ion_Measurements.png')

    # Get the f0_table variables that are needed for ALPS.
    # Protons.
    P_ParaProton = (Protons_FieldAligned.grid['Vpara'] * mp / (V_Alfven * mp))
    P_PerpProton = (np.sqrt(Protons_FieldAligned.grid['Vperp1'] ** 2 + Protons_FieldAligned.grid['Vperp2'] ** 2) * mp / (V_Alfven * mp))
    vdf_proton = Protons_FieldAligned.get_vdf().astype(np.float64)
    VDF_Proton = (vdf_proton * (mp ** (-3))) / ((mp * V_Alfven) ** -3 * N_proton)

    # Alphas.
    P_ParaAlpha = (Alphas_FieldAligned.grid['Vpara'] * ma / (V_Alfven * mp))
    P_PerpAlpha = (np.sqrt(Alphas_FieldAligned.grid['Vperp1'] ** 2 + Alphas_FieldAligned.grid['Vperp2'] ** 2) * ma / (V_Alfven * mp))
    vdf_alpha = Alphas_FieldAligned.get_vdf().astype(np.float64)
    VDF_Alpha = (vdf_alpha * (ma **(-3))) / ((mp * V_Alfven) ** -3 * N_alpha)

    # Filter points where VDF > 0 for protons
    mask_proton = VDF_Proton > 0
    P_ParaProton = P_ParaProton[mask_proton]
    P_PerpProton = P_PerpProton[mask_proton]
    VDF_Proton = VDF_Proton[mask_proton]
    # Filter points where VDF > 0 for alphas
    mask_alpha = VDF_Alpha > 0
    P_ParaAlpha = P_ParaAlpha[mask_alpha]
    P_PerpAlpha = P_PerpAlpha[mask_alpha]
    VDF_Alpha = VDF_Alpha[mask_alpha]

    # f0-tables.
    f0_table_proton = np.array([[P_PerpProton[i], P_ParaProton[i], VDF_Proton[i]] for i in range(len(P_ParaProton))])
    f0_table_alpha = np.array([[P_PerpAlpha[i], P_ParaAlpha[i], VDF_Alpha[i]] for i in range(len(P_ParaAlpha))])

    # Write these into an .array file.
    # Particle species 1: Proton
    Proton_fname = f"test_SO_{yymmdd}_{hhmmss_str}.1.array"
    if os.path.exists(res_dir + Proton_fname):
        os.remove(res_dir + Proton_fname)
    np.savetxt(res_dir + Proton_fname, f0_table_proton, fmt='%.18e')

    # Particle species 2: Alpha
    Alpha_fname = f"test_SO_{yymmdd}_{hhmmss_str}.2.array"
    if os.path.exists(res_dir + Alpha_fname):
        os.remove(res_dir + Alpha_fname)
    np.savetxt(res_dir + Alpha_fname, f0_table_alpha, fmt='%.18e')


    # Step 2:
    # ===========================================================================
    # All saved! Read back and carry on!
    with open(res_dir + Proton_fname, 'rb') as f:
        lines = f.readlines()
        f0_table_proton = [list(map(float, line.split())) for line in lines]

    p_perp_proton = np.array([line[0] for line in f0_table_proton])
    p_para_proton = np.array([line[1] for line in f0_table_proton])
    f0_VDF_proton = np.array([line[2] for line in f0_table_proton])

    v_perp_proton = p_perp_proton * (V_Alfven * mp) / (mp)
    v_para_proton = p_para_proton * (V_Alfven * mp) / (mp)
    vdf_proton = f0_VDF_proton * ((mp * V_Alfven) ** -3 * N_proton) / (mp ** -3)

    with open(res_dir + Alpha_fname, 'rb') as f:
        lines = f.readlines()
        f0_table_alpha = [list(map(float, line.split())) for line in lines]

    p_perp_alpha = np.array([line[0] for line in f0_table_alpha])
    p_para_alpha = np.array([line[1] for line in f0_table_alpha])
    f0_VDF_alpha = np.array([line[2] for line in f0_table_alpha])

    v_perp_alpha = p_perp_alpha * (V_Alfven * mp) / (ma)
    v_para_alpha = p_para_alpha * (V_Alfven * mp) / (ma)
    vdf_alpha = f0_VDF_alpha * ((mp * V_Alfven) ** -3 * N_alpha) / (ma ** -3)

    # ===============================================================================
    # Part where we need to manully set box.
    # Great! Now, let's create the collared distribution!
    # First of all, define a box where the inside points well be kept.
    vpara_p_lower = - 1.5 * V_Alfven
    vpara_p_upper = 2.5 * V_Alfven
    vperp_p_upper = 3.0 * V_Alfven

    where_p = np.where((v_para_proton > vpara_p_lower) & (v_para_proton < vpara_p_upper) & (v_perp_proton < vperp_p_upper))
    v_para_proton = v_para_proton[where_p]
    v_perp_proton = v_perp_proton[where_p]
    vdf_proton = vdf_proton[where_p]

    # Alphas 
    vpara_a_lower = - 1.0 * V_Alfven
    vpara_a_upper = 4.0 * V_Alfven
    vperp_a_upper = 2.0 * V_Alfven

    where_a = np.where((v_para_alpha > vpara_a_lower) & (v_para_alpha < vpara_a_upper) & (v_perp_alpha < vperp_a_upper))
    v_para_alpha = v_para_alpha[where_a]
    v_perp_alpha = v_perp_alpha[where_a]
    vdf_alpha = vdf_alpha[where_a]

    # get a bi-Maxwellian distribution with a larger area.
    # bm here means bi-Maxwellian.
    # proton
    vpar_bm_p = np.linspace(-3.01 * V_Alfven, 3.01 * V_Alfven, n_para + 1)
    vper_bm_p = np.linspace(0, 3.01 * V_Alfven, n_perp + 1)
    vpar_grid_bm_p, vper_grid_bm_p = np.meshgrid(vpar_bm_p, vper_bm_p)

    biMax_proton = bi_maxwellian(vpar_grid_bm_p, vper_grid_bm_p, N_proton, Tp_para, Tp_perp, mp)

    # alpha
    vpar_bm_a = np.linspace(-2.01 * V_Alfven, 4.01 * V_Alfven, n_para + 1)
    vper_bm_a = np.linspace(0, 3.01 * V_Alfven, n_perp + 1)
    vpar_grid_bm_a, vper_grid_bm_a = np.meshgrid(vpar_bm_a, vper_bm_a)
    
    biMax_alpha = bi_maxwellian(vpar_grid_bm_a, vper_grid_bm_a, N_alpha, Ta_para, Ta_perp, ma, drift_par=ua_drift)

    # ================================================================================

    # Get a colorbar norm
    vmin_p = np.log10(np.min(biMax_proton[biMax_proton > 1e-14]))
    vmax_p = np.log10(np.max(biMax_proton))
    norm_p = colors.Normalize(vmin=vmin_p, vmax=vmax_p)
    cmap = plt.get_cmap('viridis')

    # Plot the measurements over the bi-Maxwellian distribution.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Get a colorbar norm
    vmin_p = np.log10(np.min(biMax_proton[biMax_proton > 1e-14]))
    vmax_p = np.log10(np.max(biMax_proton))
    norm_p = colors.Normalize(vmin=vmin_p, vmax=vmax_p)
    cmap = plt.get_cmap('viridis')

    cs1 = ax1.contourf(vpar_grid_bm_p / V_Alfven, vper_grid_bm_p / V_Alfven, np.log10(biMax_proton), levels=100, alpha=0.4, cmap=cmap, norm=norm_p)
    ax1.scatter(v_para_proton / V_Alfven, v_perp_proton / V_Alfven, c=np.log10(vdf_proton), s=5.0, cmap=cmap, norm=norm_p)
    cbar1 = fig.colorbar(cs1, ax=ax1)
    ax1.set_xlabel('$v_{\parallel}$ / $V_A$')
    ax1.set_ylabel('$v_{\perp}$ / $V_A$')
    ax1.set_title('Bi-Maxwellian Proton VDF')
    ax1.set_aspect('equal')

    # Get a colorbar norm
    vmin_a = np.log10(np.min(biMax_alpha[biMax_alpha > 1e-14]))
    vmax_a = np.log10(np.max(biMax_alpha))
    norm_a = colors.Normalize(vmin=vmin_a, vmax=vmax_a)
    cmap = plt.get_cmap('viridis')

    cs2 = ax2.contourf(vpar_grid_bm_a / V_Alfven, vper_grid_bm_a / V_Alfven, np.log10(biMax_alpha), levels=100, alpha=0.4, cmap=cmap, norm=norm_a)
    ax2.scatter(v_para_alpha / V_Alfven, v_perp_alpha / V_Alfven, c=np.log10(vdf_alpha), s=5.0, cmap=cmap, norm=norm_a)
    cbar2 = fig.colorbar(cs2, ax=ax2)
    ax2.set_xlabel('$v_{\parallel}$ / $V_A$')
    ax2.set_ylabel('$v_{\perp}$ / $V_A$')
    ax2.set_title('Bi-Maxwellian Alpha VDF')
    ax2.set_aspect('equal')
    plt.savefig(res_dir + 'Measurements_over_BiMaxwellian.png')

    # Let's mirror it also into the negative v_perp space.
    # Mirror to the - v_perp plane
    # Plot the measurements over the bi-Maxwellian distribution.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Get a colorbar norm
    vmin_p = np.log10(np.min(biMax_proton[biMax_proton > 1e-14]))
    vmax_p = np.log10(np.max(biMax_proton))
    norm_p = colors.Normalize(vmin=vmin_p, vmax=vmax_p)
    cmap = plt.get_cmap('viridis')

    cs1 = ax1.contourf(vpar_grid_bm_p / V_Alfven, vper_grid_bm_p / V_Alfven, np.log10(biMax_proton), levels=100, alpha=0.4, cmap=cmap, norm=norm_p)
    ax1.contourf(vpar_grid_bm_p / V_Alfven, -vper_grid_bm_p / V_Alfven, np.log10(biMax_proton), levels=100, alpha=0.4, cmap=cmap, norm=norm_p)
    ax1.scatter(v_para_proton / V_Alfven, v_perp_proton / V_Alfven, c=np.log10(vdf_proton), s=5.0, cmap=cmap, norm=norm_p)
    ax1.scatter(v_para_proton / V_Alfven, -v_perp_proton / V_Alfven, c=np.log10(vdf_proton), s=5.0, cmap=cmap, norm=norm_p)
    cbar1 = fig.colorbar(cs1, ax=ax1)
    ax1.set_xlabel('$v_{\parallel}$ / $V_A$')
    ax1.set_ylabel('$v_{\perp}$ / $V_A$')
    ax1.set_title('Bi-Maxwellian Proton VDF')
    ax1.set_aspect('equal')

    # Get a colorbar norm
    vmin_a = np.log10(np.min(biMax_alpha[biMax_alpha > 1e-14]))
    vmax_a = np.log10(np.max(biMax_alpha))
    norm_a = colors.Normalize(vmin=vmin_a, vmax=vmax_a)
    cmap = plt.get_cmap('viridis')

    cs2 = ax2.contourf(vpar_grid_bm_a / V_Alfven, vper_grid_bm_a / V_Alfven, np.log10(biMax_alpha), levels=100, alpha=0.4, cmap=cmap, norm=norm_a)
    ax2.contourf(vpar_grid_bm_a / V_Alfven, -vper_grid_bm_a / V_Alfven, np.log10(biMax_alpha), levels=100, alpha=0.4, cmap=cmap, norm=norm_a)
    ax2.scatter(v_para_alpha / V_Alfven, v_perp_alpha / V_Alfven, c=np.log10(vdf_alpha), s=5.0, cmap=cmap, norm=norm_a)
    ax2.scatter(v_para_alpha / V_Alfven, -v_perp_alpha / V_Alfven, c=np.log10(vdf_alpha), s=5.0, cmap=cmap, norm=norm_a)
    cbar2 = fig.colorbar(cs2, ax=ax2)
    ax2.set_xlabel('$v_{\parallel}$ / $V_A$')
    ax2.set_ylabel('$v_{\perp}$ / $V_A$')
    ax2.set_title('Bi-Maxwellian Alpha VDF')
    ax2.set_aspect('equal')
    plt.savefig(res_dir + 'Measurements_over_BiMaxwellian_mirrored.png')

    # Great! Now let's get the mask (convex hull) to include all the measuring points.
    mask_proton = generate_mask_from_points(v_para_proton, v_perp_proton, vpar_grid_bm_p, vper_grid_bm_p)
    mask_alpha = generate_mask_from_points(v_para_alpha, v_perp_alpha, vpar_grid_bm_a, vper_grid_bm_a)

    # Build symmetric input for protons
    points_proton_sym = np.vstack([
        np.column_stack((v_para_proton, v_perp_proton)),
        np.column_stack((v_para_proton, -v_perp_proton))
    ])
    values_proton_sym = np.concatenate([vdf_proton, vdf_proton])

    # Build symmetric input for alphas
    points_alpha_sym = np.vstack([
        np.column_stack((v_para_alpha, v_perp_alpha)),
        np.column_stack((v_para_alpha, -v_perp_alpha))
    ])
    values_alpha_sym = np.concatenate([vdf_alpha, vdf_alpha])

    # Interpolate for protons
    vdf_linear_sym = griddata(points_proton_sym, values_proton_sym, (vpar_grid_bm_p, vper_grid_bm_p), method='linear')
    vdf_nearest_sym = griddata(points_proton_sym, values_proton_sym, (vpar_grid_bm_p, vper_grid_bm_p), method='nearest')
    vdf_combined_sym = np.where(np.isnan(vdf_linear_sym), vdf_nearest_sym, vdf_linear_sym)
    #vdf_smooth_sym = gaussian_filter(vdf_linear_sym, sigma=1)
    vdf_smooth_sym = gaussian_filter(vdf_combined_sym, sigma=1)

    # Interpolate for alphas
    vdf_linear_alpha_sym = griddata(points_alpha_sym, values_alpha_sym, (vpar_grid_bm_a, vper_grid_bm_a), method='linear')
    vdf_nearest_alpha_sym = griddata(points_alpha_sym, values_alpha_sym, (vpar_grid_bm_a, vper_grid_bm_a), method='nearest')
    vdf_combined_alpha_sym = np.where(np.isnan(vdf_linear_alpha_sym), vdf_nearest_alpha_sym, vdf_linear_alpha_sym)
    #vdf_smooth_alpha_sym = gaussian_filter(vdf_linear_alpha_sym, sigma=1)
    vdf_smooth_alpha_sym = gaussian_filter(vdf_combined_alpha_sym, sigma=1)

    # Everything else is the same as before!
    # Use your existing mask (for v_perp > 0), smooth it, and blend:
    smooth_weight_proton = smooth_mask(mask_proton, sigma=smooth_sigma)
    vdf_final_proton = merge_vdf_logspace(vdf_smooth_sym, biMax_proton, smooth_weight_proton)
    vdf_final_proton = renormalize_vdf(vdf_final_proton, vpar_grid_bm_p, vper_grid_bm_p, N_proton)

    smooth_weight_alpha = smooth_mask(mask_alpha, sigma=smooth_sigma)
    vdf_final_alpha = merge_vdf_logspace(vdf_smooth_alpha_sym, biMax_alpha, smooth_weight_alpha)
    vdf_final_alpha = renormalize_vdf(vdf_final_alpha, vpar_grid_bm_a, vper_grid_bm_a, N_alpha) 

    # plot the masks, so we keep track of them.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.contourf(vpar_grid_bm_p / V_Alfven, vper_grid_bm_p / V_Alfven, mask_proton, levels=1)
    ax1.set_title("Proton Mask")
    ax1.set_xlabel('$v_{\parallel}$ / $V_A$')
    ax1.set_ylabel('$v_{\perp}$ / $V_A$')
    ax1.set_aspect('equal')

    cs = ax2.contourf(vpar_grid_bm_a / V_Alfven, vper_grid_bm_a / V_Alfven, mask_alpha, levels=1)
    ax2.set_title("Alpha Mask")
    ax2.set_xlabel('$v_{\parallel}$ / $V_A$')
    ax2.set_ylabel('$v_{\perp}$ / $V_A$')
    cbar = fig.colorbar(cs, ax=ax2)
    ax2.set_aspect('equal')
    plt.savefig(res_dir + 'Masks.png')

    # plot the smoothed mask! Keep every step in mind!
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.contourf(vpar_grid_bm_p / V_Alfven, vper_grid_bm_p / V_Alfven, smooth_weight_proton, levels=100)
    ax1.set_xlabel('$v_{\parallel}$ / $V_A$')
    ax1.set_ylabel('$v_{\perp}$ / $V_A$')
    ax1.set_title("Proton Smooth Mask")

    cs = ax2.contourf(vpar_grid_bm_a / V_Alfven, vper_grid_bm_a / V_Alfven, smooth_weight_alpha, levels=100)
    ax2.set_xlabel('$v_{\parallel}$ / $V_A$')
    ax2.set_ylabel('$v_{\perp}$ / $V_A$')
    ax2.set_title("Alpha Smooth Mask")
    cbar = fig.colorbar(cs, ax=ax2)
    plt.savefig(res_dir + 'Smooth_Masks.png')


    # Plot the symmetric interpolation result (before blending), with mask as contours
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cf1 = ax1.contourf(vpar_grid_bm_p / V_Alfven, vper_grid_bm_p / V_Alfven, np.log10(vdf_smooth_sym), levels=30)
    ax1.contour(vpar_grid_bm_p / V_Alfven, vper_grid_bm_p / V_Alfven, mask_proton, levels=[0.5], colors='r', linewidths=2)
    ax1.set_xlabel(r"$v_\parallel$ / $V_A$")
    ax1.set_ylabel(r"$v_\perp$ / $V_A$")
    ax1.set_title("Interpolated Proton VDF (Symmetric)")
    ax1.set_aspect('equal')
    fig.colorbar(cf1, ax=ax1)

    cf2 = ax2.contourf(vpar_grid_bm_a / V_Alfven, vper_grid_bm_a / V_Alfven, np.log10(vdf_smooth_alpha_sym), levels=30)
    ax2.contour(vpar_grid_bm_a / V_Alfven, vper_grid_bm_a / V_Alfven, mask_alpha, levels=[0.5], colors='r', linewidths=2)
    ax2.set_xlabel(r"$v_\parallel$ / $V_A$")
    ax2.set_ylabel(r"$v_\perp$ / $V_A$")
    ax2.set_title("Interpolated Alpha VDF (Symmetric)")
    ax2.set_aspect('equal')
    fig.colorbar(cf2, ax=ax2)
    plt.savefig(res_dir + 'Symmetric_Interpolation.png')

    # Real-space Gaussian smoothing
    sigma_proton = 2  # Tune this based on your grid and desired scale
    sigma_alpha = 2

    vdf_final_proton = gaussian_filter(vdf_final_proton, sigma=sigma_proton)
    vdf_final_alpha = gaussian_filter(vdf_final_alpha, sigma=sigma_alpha)

    # Clip to avoid log10 errors
    vdf_final_proton = np.clip(vdf_final_proton, 1e-30, None)
    vdf_final_alpha = np.clip(vdf_final_alpha, 1e-30, None)

    # Now! plot the final vdf!
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cs1 = ax1.contourf(vpar_grid_bm_p / V_Alfven, vper_grid_bm_p / V_Alfven, np.log10(vdf_final_proton), levels=30)
    ax1.set_xlabel(r"$v_\parallel$ / $V_A$")
    ax1.set_ylabel(r"$v_\perp$ / $V_A$")
    ax1.set_title("Final Proton VDF")
    cbar1 = fig.colorbar(cs1, ax=ax1)
    ax1.set_aspect('equal')

    cs2 = ax2.contourf(vpar_grid_bm_a / V_Alfven, vper_grid_bm_a / V_Alfven, np.log10(vdf_final_alpha), levels=30)
    ax2.set_xlabel(r"$v_\parallel$ / $V_A$")
    ax2.set_ylabel(r"$v_\perp$ / $V_A$")
    ax2.set_title("Final Alpha VDF")
    cbar2 = fig.colorbar(cs2, ax=ax2)
    ax2.set_aspect('equal')
    plt.savefig(res_dir + 'Final_VDFs.png')


    # Let's also plot: proton_vdf + biMax_alpha; biMax_proton + alpha_vdf, biMax_proton + biMax_alpha.
    # 1 - proton_vdf + biMax_alpha
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cs1 = ax1.contourf(vpar_grid_bm_p / V_Alfven, vper_grid_bm_p / V_Alfven, np.log10(vdf_final_proton), levels=30)
    ax1.set_xlabel(r"$v_\parallel$ / $V_A$")
    ax1.set_ylabel(r"$v_\perp$ / $V_A$")
    ax1.set_title("Proton VDF")
    cbar1 = fig.colorbar(cs1, ax=ax1)
    ax1.set_aspect('equal')

    cs2 = ax2.contourf(vpar_grid_bm_a / V_Alfven, vper_grid_bm_a / V_Alfven, np.log10(biMax_alpha), levels=30)
    ax2.set_xlabel(r"$v_\parallel$ / $V_A$")
    ax2.set_ylabel(r"$v_\perp$ / $V_A$")
    ax2.set_title("Bi-Maxwellian Alpha VDF")
    cbar2 = fig.colorbar(cs2, ax=ax2)
    ax2.set_aspect('equal')
    plt.savefig(res_dir + 'VDFProton_BiMaxAlpha.png')

    # 2 - biMax_proton + alpha_vdf
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cs1 = ax1.contourf(vpar_grid_bm_p / V_Alfven, vper_grid_bm_p / V_Alfven, np.log10(biMax_proton), levels=30)
    ax1.set_xlabel(r"$v_\parallel$ / $V_A$")
    ax1.set_ylabel(r"$v_\perp$ / $V_A$")
    ax1.set_title("Bi-Maxwellian Proton VDF")
    cbar1 = fig.colorbar(cs1, ax=ax1)
    ax1.set_aspect('equal')

    cs2 = ax2.contourf(vpar_grid_bm_a / V_Alfven, vper_grid_bm_a / V_Alfven, np.log10(vdf_final_alpha), levels=30)
    ax2.set_xlabel(r"$v_\parallel$ / $V_A$")
    ax2.set_ylabel(r"$v_\perp$ / $V_A$")
    ax2.set_title("Alpha VDF")
    cbar2 = fig.colorbar(cs2, ax=ax2)
    ax2.set_aspect('equal')
    plt.savefig(res_dir + 'BiMaxProton_VDFAlpha.png')

    # 3 - biMax_proton + biMax_alpha
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cs1 = ax1.contourf(vpar_grid_bm_p / V_Alfven, vper_grid_bm_p / V_Alfven, np.log10(biMax_proton), levels=30)
    ax1.set_xlabel(r"$v_\parallel$ / $V_A$")
    ax1.set_ylabel(r"$v_\perp$ / $V_A$")
    ax1.set_title("Bi-Maxwellian Proton VDF")
    cbar1 = fig.colorbar(cs1, ax=ax1)
    ax1.set_aspect('equal')
    cs2 = ax2.contourf(vpar_grid_bm_a / V_Alfven, vper_grid_bm_a / V_Alfven, np.log10(biMax_alpha), levels=30)
    ax2.set_xlabel(r"$v_\parallel$ / $V_A$")
    ax2.set_ylabel(r"$v_\perp$ / $V_A$")
    ax2.set_title("Bi-Maxwellian Alpha VDF")
    cbar2 = fig.colorbar(cs2, ax=ax2)
    ax2.set_aspect('equal')
    plt.savefig(res_dir + 'BiMaxProton_BiMaxAlpha.png')

    # After this interpolation, some moments might have changed. We need to calculate them and, let's save them!
    delta_v_para_p = vpar_grid_bm_p[0, 1] - vpar_grid_bm_p[0, 0]
    delta_v_perp_p = vper_grid_bm_p[1, 0] - vper_grid_bm_p[0, 0]

    delta_v_para_a = vpar_grid_bm_a[0, 1] - vpar_grid_bm_a[0, 0]
    delta_v_perp_a = vper_grid_bm_a[1, 0] - vper_grid_bm_a[0, 0]

    # densities.
    density_proton = 2 * np.pi * np.sum(vdf_final_proton * vper_grid_bm_p) * delta_v_para_p * delta_v_perp_p
    density_alpha = 2 * np.pi * np.sum(vdf_final_alpha * vper_grid_bm_a) * delta_v_para_a * delta_v_perp_a
    
    # velocities
    # Because the vdf is shifted, so do not be bothered by the differences in the velocities too much.
    # Just make sure that the drift velocity matches.
    u_para_alpha = (2 * np.pi / density_alpha) * np.sum(vdf_final_alpha * vper_grid_bm_a * vpar_grid_bm_a) * delta_v_para_a * delta_v_perp_a
    u_para_proton = (2 * np.pi / density_proton) * np.sum(vdf_final_proton * vper_grid_bm_p * vpar_grid_bm_p) * delta_v_para_p * delta_v_perp_p

    u_center_of_mass = (N_proton * u_para_proton + N_alpha * u_para_alpha) / (N_proton + N_alpha)

    # temperatures
    # Protons
    # Parallel temperature (eV)
    Tp_para_eV = (2 * np.pi / density_proton) * np.sum(
        vdf_final_proton * vper_grid_bm_p * (vpar_grid_bm_p - u_para_proton)**2
    ) * delta_v_para_p * delta_v_perp_p * mp / (e)

    # Perpendicular temperature (eV)
    Tp_perp_eV = (np.pi / density_proton) * np.sum(
        vdf_final_proton * vper_grid_bm_p * vper_grid_bm_p**2
    ) * delta_v_para_p * delta_v_perp_p * mp / e

    # Alphas
    # Parallel temperature (eV)
    Ta_para_eV = (2 * np.pi / density_alpha) * np.sum(
        vdf_final_alpha * vper_grid_bm_a * (vpar_grid_bm_a - u_para_alpha)**2
    ) * delta_v_para_a * delta_v_perp_a * ma / e
    # Perpendicular temperature (eV)
    Ta_perp_eV = (np.pi / density_alpha) * np.sum(
        vdf_final_alpha * vper_grid_bm_a * vper_grid_bm_a**2
    ) * delta_v_para_a * delta_v_perp_a * ma / e

    # Shift to the center of mass frame
    vpar_grid_bm_p = vpar_grid_bm_p - u_center_of_mass
    vpar_grid_bm_a = vpar_grid_bm_a - u_center_of_mass

    u_para_electron = (N_proton * u_para_proton + 2 * N_alpha * u_para_alpha) / (N_proton + 2 * N_alpha)

    # All bulk velocities in the center of mass frame.
    u_para_proton -= u_center_of_mass
    u_para_alpha -= u_center_of_mass
    u_para_electron -= u_center_of_mass

    parameters['N_proton_collared'] = density_proton
    parameters['N_alpha_collared'] = density_alpha
    parameters['u_para_proton_collared'] = u_para_proton
    parameters['u_para_alpha_collared'] = u_para_alpha
    parameters['u_para_electron_collared'] = u_para_electron
    parameters['Tp_para_collared'] = Tp_para_eV
    parameters['Tp_perp_collared'] = Tp_perp_eV
    parameters['Ta_para_collared'] = Ta_para_eV
    parameters['Ta_perp_collared'] = Ta_perp_eV

    parameters_df = pd.DataFrame(parameters, index=[0])
    parameters_df.to_csv(res_dir + 'parameters.csv', index=False)

    # Great! Now transfer data into an f0 table.
    # proton
    vpar_grid_p_flat = vpar_grid_bm_p.flatten()
    vper_grid_p_flat = vper_grid_bm_p.flatten()
    vdf_final_flat_proton = vdf_final_proton.flatten()

    pper_grid_flat_proton = vper_grid_p_flat * (mp) / (V_Alfven * mp)
    ppar_grid_flat_proton = vpar_grid_p_flat * (mp) / (V_Alfven * mp)
    f0_vdf_proton = vdf_final_flat_proton * (mp ** -3) / ((mp * V_Alfven) ** -3 * N_proton)

    # alpha
    vpar_grid_a_flat = vpar_grid_bm_a.flatten()
    vper_grid_a_flat = vper_grid_bm_a.flatten()
    vdf_fina_flat_alpha = vdf_final_alpha.flatten()

    pper_grid_flat_alpha = vper_grid_a_flat * (ma) / (V_Alfven * mp)
    ppar_grid_flat_alpha = vpar_grid_a_flat * (ma) / (V_Alfven * mp)
    f0_vdf_alpha = vdf_fina_flat_alpha * (ma ** -3) / ((mp * V_Alfven) ** -3 * N_alpha)

    # generate the f0 tables.
    f0_table_proton = np.array([[pper_grid_flat_proton[i], ppar_grid_flat_proton[i], f0_vdf_proton[i]] for i in range(len(pper_grid_flat_proton))])
    f0_table_alpha = np.array([[pper_grid_flat_alpha[i], ppar_grid_flat_alpha[i], f0_vdf_alpha[i]] for i in range(len(pper_grid_flat_alpha))])

    # save them!
    np.savetxt(res_dir + f"test_SO_{yymmdd}_{hhmmss_str}_collared.1.array", f0_table_proton, fmt='%.18e')
    np.savetxt(res_dir + f"test_SO_{yymmdd}_{hhmmss_str}_collared.2.array", f0_table_alpha, fmt='%.18e')

    # Also, let's transfer the biMax VDFs into f0 tables as well, for analysis.
    biMax_proton_flat = biMax_proton.flatten()
    biMax_alpha_flat = biMax_alpha.flatten()

    f0_biMax_vdf_proton = biMax_proton_flat * (mp ** -3) / ((mp * V_Alfven) ** -3 * N_proton)
    f0_biMax_vdf_alpha = biMax_alpha_flat * (ma ** -3) / ((mp * V_Alfven) ** -3 * N_alpha)

    f0_table_biMax_proton = np.array([[pper_grid_flat_proton[i], ppar_grid_flat_proton[i], f0_biMax_vdf_proton[i]] for i in range(len(pper_grid_flat_proton))])
    f0_table_biMax_alpha = np.array([[pper_grid_flat_alpha[i], ppar_grid_flat_alpha[i], f0_biMax_vdf_alpha[i]] for i in range(len(pper_grid_flat_alpha))])
    
    np.savetxt(res_dir + f"test_SO_{yymmdd}_{hhmmss_str}_biMax.1.array", f0_table_biMax_proton, fmt='%.18e')
    np.savetxt(res_dir + f"test_SO_{yymmdd}_{hhmmss_str}_biMax.2.array", f0_table_biMax_alpha, fmt='%.18e')

    return 0



if __name__ == "__main__":
    # run the main function
    main()


