"""
This is the code for reading EAS1 and EAS2 data.

====================================================================================================
Several steps are involved:
1. Remove one-particle noise level.
2. Remove the low-energy range, which is highly contaminated by photo- and secondary electrons.
   We use a bi-Maxwellian model to fit the core range, then apply this fitted bi-Maxwellian function to the low-energy range.

Results are stored as:

Electron_Spherical: VDFs in the (elevation, azimuth, velocity) coordinate.
Electron_3D_FieldAligned: VDFs in the (Vpara, Vperp1, Vperp2) coordinate.

====================================================================================================

Author = Hao Ran    GitHub: #RanHao1999       Mail: hao.ran.24@ucl.ac.uk
Date = 2025-09-26
Mullard Space Science Laboratory, UCL
"""

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from spacepy import pycdf
from datetime import datetime, timedelta
import bisect
from scipy.interpolate import griddata
import matplotlib.colors as colors
from scipy.optimize import least_squares
import csv  # Add this import at the top of the file

os.chdir(sys.path[0])

# These are the functions that I write.
from Funcs import *
from SolarWindPack import *

# The function for calculating density
# We need to verify!
def cal_density(vdf, azimuth, elevation, velocity):

    subazi, subele, subspeed = np.meshgrid(azimuth, elevation, velocity, indexing='ij')
    # With a Jacobian matrix v^2 cos(elevation)
    int_kernal = vdf * (subspeed ** 2) * np.cos(np.radians(subele))
    int_ele = np.trapezoid(int_kernal, x=np.radians(elevation), axis=1)
    int_ele_energy = np.trapezoid(int_ele, x=velocity, axis=1)
    int_all = np.trapezoid(int_ele_energy, x=np.radians(azimuth), axis=0)

    return np.abs(float(int_all))

# Read Data in this box.
def readCDFs(filepaths, keys):
    # In this function, we read the data of several CDF files and concatenate them into one array.
    # The variables in 'keys' will be subtracted.
    # The function returns the concatenated array.

    DataSet_list = [] # Every elements contains the data_arrays of one key.

    for key in keys:
        data_arrays = []
        for path in filepaths:
            cdf = pycdf.CDF(path)
            data_arrays.append(cdf[key][...])
            cdf.close()
        DataSet_list.append(data_arrays)

    Output_list = []
    for Data in DataSet_list:
        output = np.concatenate(Data, axis=0)
        Output_list.append(output)

    return Output_list

def get_common_timestamps(EAS1_Epoch, EAS2_Epoch):
    # EAS1 and EAS2 might not be entirely matched.
    # Let's keep only the intersected timestamps.
    # Precision to "Seconds".

    # Convert to numpy datetime64 with second precision (ignoring microseconds)
    EAS1_Seconds = EAS1_Epoch.astype('datetime64[s]')
    EAS2_Seconds = EAS2_Epoch.astype('datetime64[s]')

    # Find common timestamps (ignoring microseconds)
    common_seconds = np.intersect1d(EAS1_Seconds, EAS2_Seconds)

    # Get indices of common timestamps in original arrays
    indices_EAS1 = np.nonzero(np.isin(EAS1_Seconds, common_seconds))[0]
    indices_EAS2 = np.nonzero(np.isin(EAS2_Seconds, common_seconds))[0]

    return indices_EAS1, indices_EAS2

def one_particle_noise_level(PSD_Data, Count_Data):
    # get the one-particle noise level.

    # Test if PSD_Data and Count_Data have the same shape.
    assert PSD_Data.shape == Count_Data.shape, 'PSD_Data and Count_Data have different shapes.'

    # Calculate the one-particle noise level.
    noise_level = np.divide(PSD_Data, Count_Data, out=np.zeros_like(PSD_Data, dtype=float), where=Count_Data!=0)
    noise_level[np.isnan(noise_level)] = 0  # Drop the Nan values and replace them with 0.
    one_particle_noise_level = np.nanpercentile(noise_level, 99.99, axis=0)

    return one_particle_noise_level

def energy2velocity(E):
    me = 9.10938356e-31 # kg
    energy_Joule = E * 1.60218e-19 # eV to Joule
    velocity = np.sqrt(2 * energy_Joule / me) # m/s
    return velocity

def FindIndexinInterval(tstart, tend, epoch):
    # Given the epoch of the data, find all the indexes within the given interval.
    if tstart > tend:
        raise ValueError("tstart should be smaller than tend.")
    result = [[idx, epoch[idx]] for idx in range(len(epoch)) if tstart <= epoch[idx] <= tend]
    return result

def GetVelocityVector(theta, phi, velocity):
    theta_grid, velocity_grid, phi_grid = np.meshgrid(np.radians(theta), velocity,  np.radians(phi), indexing='ij')
    vx = velocity_grid * np.cos(theta_grid) * np.cos(phi_grid)
    vy = velocity_grid * np.cos(theta_grid) * np.sin(phi_grid)
    vz = - velocity_grid * np.sin(theta_grid)
    return vx, vy, vz

def Transfer2SRF(transferrMatrixx, Vx, Vy, Vz):
    Vx_SRF = transferrMatrixx[0, 0] * Vx + transferrMatrixx[0, 1] * Vy + transferrMatrixx[0, 2] * Vz
    Vy_SRF = transferrMatrixx[1, 0] * Vx + transferrMatrixx[1, 1] * Vy + transferrMatrixx[1, 2] * Vz
    Vz_SRF = transferrMatrixx[2, 0] * Vx + transferrMatrixx[2, 1] * Vy + transferrMatrixx[2, 2] * Vz
    return Vx_SRF, Vy_SRF, Vz_SRF

# Get in parallel and perpenducular plane.
def interpolate2D(Vpara, Vperp, VDF):
    # Original measurements are in 3D ()
    # Flat the 3D array to 1D array
    Vpara_flat = Vpara.flatten()
    Vperp_flat = Vperp.flatten()
    VDF_flat = VDF.flatten()

    vpar_min, vpar_max = Vpara.min(), Vpara.max()
    vperp_min, vperp_max = Vperp.min(), Vperp.max()

    vpar_bins = np.linspace(vpar_min, vpar_max, 50)
    vperp_bins = np.linspace(vperp_min, vperp_max, 25)
    vpar_grid, vperp_grid = np.meshgrid(vpar_bins, vperp_bins, indexing='ij')

    # Regrid
    H, Xedges, Yedges = np.histogram2d(Vpara_flat, Vperp_flat, bins=(vpar_bins, vperp_bins), weights=VDF_flat)
    counts, _, _ = np.histogram2d(Vpara_flat, Vperp_flat, bins=(vpar_bins, vperp_bins)) 
    H_avg = np.divide(H, counts, out=np.zeros_like(H), where=counts != 0)

    # Inteprolate
    points = np.column_stack((Vpara_flat, Vperp_flat))
    vdf_2d = griddata(points, VDF_flat, (vpar_grid, vperp_grid), method='linear')
    vdf_2d = np.nan_to_num(vdf_2d)  # For moment calculation, nan should be 0.

    return Xedges, Yedges, H_avg

def bi_Maxwellian_2D(Vpara, Vperp, n, Tpara, Tperp, V_drift):
    """
    Vpara: Parallel velocity component (m/s)
    Vperp: Perpendicular velocity component (m/s)
    n: Number density (m^-3)
    Tpara: Parallel temperature (eV)
    Tperp: Perpendicular temperature (eV)
    V_drift: Drift velocity (m/s)
    """
    # Physical constants (SI units)
    kB = 1.380649e-23       # Boltzmann constant [J/K]
    m_e = 9.10938356e-31    # Electron mass [kg]
    q_e = 1.602176634e-19   # Elementary charge [C]

    # eV to J
    Tpara_J = Tpara * q_e
    Tperp_J = Tperp * q_e

    # Thermal Velocities:
    vth_para = np.sqrt(2.0 * Tpara_J / m_e)
    vth_perp = np.sqrt(2.0 * Tperp_J / m_e)

    # Prefactor
    A = n / (np.pi ** (1.5) * vth_para * vth_perp ** 2)

    # Exponential arguement
    exponent = -((Vpara - V_drift) ** 2) / (vth_para ** 2) - (Vperp ** 2) / (vth_perp ** 2)

    return A * np.exp(exponent)

def fit_bi_Maxwellian(PSDslice, Vpara, Vperp):
    Vpara_flat = Vpara.flatten()
    Vperp_flat = Vperp.flatten()
    PSDslice_flat = PSDslice.flatten()

    def residuals(params, vpar, vperp, psd):
        n, Tpara_eV, Tperp_eV, v_drift = params
        model = bi_Maxwellian_2D(vpar, vperp, n, Tpara_eV, Tperp_eV, v_drift)
        return np.log10(model) - np.log10(psd)
    
    # Initial guess
    n0_guess = 10e6
    Tpara0_guess = 30.0
    Tperp_guess = 20.0
    v_drift_guess = 0.0
    params0 = np.array([n0_guess, Tpara0_guess, Tperp_guess, v_drift_guess])

    result = least_squares(residuals, params0, args=(Vpara_flat, Vperp_flat, PSDslice_flat))
    n_fit, Tpara_fit, Tperp_fit, v_drift_fit = result.x

    params_fit = {
        'n': n_fit,
        'Tpara': Tpara_fit,
        'Tperp': Tperp_fit,
        'v_drift': v_drift_fit
    }
    return params_fit


def main():
    # The interval that you want.
    # Please make sure your data covers you time range.
    tstart = datetime(2023, 3, 19, 9, 0, 0)
    tend = datetime(2023, 3, 19, 15, 0, 0)

    # Date, and corresponding data directory.
    yyyymmdd = tstart.strftime('%Y%m%d')
    DataDir = 'data/SO/' + yyyymmdd + '/'

    # PSD Data Paths
    EAS1_PSD_Filepaths = sorted([DataDir + x for x in os.listdir(DataDir) if 'eas1-nm3d-psd' in x])  # Corrreted PSD files
    EAS2_PSD_Filepaths = sorted([DataDir + x for x in os.listdir(DataDir) if 'eas2-nm3d-psd' in x])  # Corrreted PSD files
    EAS1_CNT_Filepaths = sorted([DataDir + x for x in os.listdir(DataDir) if 'eas1-NM3D' in x])     # Count files, for calculating 1-particle noise.
    EAS2_CNT_Filepaths = sorted([DataDir + x for x in os.listdir(DataDir) if 'eas2-NM3D' in x])     # Count files, for calculating 1-particle noise.

    # Consistent data that needs to be concatenated.
    EAS1_PSD, EAS1_Epoch = readCDFs(EAS1_PSD_Filepaths, ['SWA_EAS1_Data', 'EPOCH'])
    EAS2_PSD, EAS2_Epoch = readCDFs(EAS2_PSD_Filepaths, ['SWA_EAS2_Data', 'EPOCH'])
    EAS1_CNT = readCDFs(EAS1_CNT_Filepaths, ['SWA_EAS1_Data'])[0]   # Count data, there is only one element, so add the [0].
    EAS2_CNT = readCDFs(EAS2_CNT_Filepaths, ['SWA_EAS2_Data'])[0]   # Count data, there is only one element, so add the [0].

    # Keep the intersected timestamps.
    # EAS1 and EAS2 Epoch might not be entirely matched (Though this happens quite not frequently).
    # We need to deal with it.
    # Get the intersected timestamps.
    indices_EAS1, indices_EAS2 = get_common_timestamps(EAS1_Epoch, EAS2_Epoch)
    EAS1_PSD = EAS1_PSD[indices_EAS1]
    EAS2_PSD = EAS2_PSD[indices_EAS2]
    EAS1_Epoch = EAS1_Epoch[indices_EAS1]
    EAS2_Epoch = EAS2_Epoch[indices_EAS2]
    EAS1_CNT = EAS1_CNT[indices_EAS1]
    EAS2_CNT = EAS2_CNT[indices_EAS2]

    # Get and remove One-particle noise level.
    #OneParticleNoiseLevel_EAS1 = one_particle_noise_level(EAS1_PSD, EAS1_CNT)
    #OneParticleNoiseLevel_EAS2 = one_particle_noise_level(EAS2_PSD, EAS2_CNT)

    # Other Parameters
    EAS1_cdffile = pycdf.CDF(EAS1_PSD_Filepaths[0])
    EAS2_cdffile = pycdf.CDF(EAS2_PSD_Filepaths[0])

    # Transfer to SRF.
    EAS1_To_SRF = EAS1_cdffile['EAS1_TO_SRF'][...]
    EAS2_To_SRF = EAS2_cdffile['EAS2_TO_SRF'][...]
    
    # Replace the filled values with 0.
    Temp_EAS1PSD = EAS1_cdffile['SWA_EAS1_Data']
    FillValues_Mask = np.where(EAS1_PSD == Temp_EAS1PSD.attrs['FILLVAL'])
    EAS1_PSD[FillValues_Mask] = 0

    Temp_EAS2PSD = EAS2_cdffile['SWA_EAS2_Data']
    FillValues_Mask = np.where(EAS2_PSD == Temp_EAS2PSD.attrs['FILLVAL'])
    EAS2_PSD[FillValues_Mask] = 0

    # Delete the temporary variables.
    del Temp_EAS1PSD, Temp_EAS2PSD

    # Coordinate for EAS1.
    energyEAS1 = EAS1_cdffile['SWA_EAS1_ENERGY'][...]
    thetaEAS1 = EAS1_cdffile['SWA_EAS_ELEVATION'][...]
    phiEAS1 = EAS1_cdffile['SWA_EAS_AZIMUTH'][...]
    # Coordinates for EAS2.
    energyEAS2 = EAS2_cdffile['SWA_EAS2_ENERGY'][...]
    thetaEAS2 = EAS2_cdffile['SWA_EAS_ELEVATION'][...]
    phiEAS2 = EAS2_cdffile['SWA_EAS_AZIMUTH'][...]

    velocityEAS1 = energy2velocity(energyEAS1)
    velocityEAS2 = energy2velocity(energyEAS2)

    # Keep only the interval that we need.
    tstart_index = bisect.bisect_left(EAS1_Epoch, tstart)
    tend_index = bisect.bisect_right(EAS1_Epoch, tend)

    EAS1_PSD = EAS1_PSD[tstart_index:tend_index]
    EAS2_PSD = EAS2_PSD[tstart_index:tend_index]
    EAS1_Epoch = EAS1_Epoch[tstart_index:tend_index]
    EAS2_Epoch = EAS2_Epoch[tstart_index:tend_index]
    EAS1_CNT = EAS1_CNT[tstart_index:tend_index]
    EAS2_CNT = EAS2_CNT[tstart_index:tend_index]

    # Read Magnetometer Data.
    mag_filename = next(f for f in os.listdir(f'data/SO/{yyyymmdd}') if 'mag-srf-normal' in f)
    mag_cdffile = pycdf.CDF(f'data/SO/{yyyymmdd}/{mag_filename}')
    Mag_Epoch = mag_cdffile['EPOCH'][...]
    B_SRF = mag_cdffile['B_SRF'][...]

    # All the indexes and times within the given interval.
    idx_times = FindIndexinInterval(tstart, tend, EAS1_Epoch)

    # Process every time slice, seperately.
    for idx_time in idx_times:
        index = idx_time[0]
        tslice = idx_time[1]
        yyyymmdd = tslice.strftime('%Y%m%d')
        hhmmss = tslice.strftime('%H%M%S')
        # Create the directory if not exists.
        SaveDir = 'result/SO/'+yyyymmdd+'/Particles/Electrons/'+hhmmss+'/'
        if not os.path.exists(SaveDir):
            os.makedirs(SaveDir)

        # Mag that Represents the time slice.
        tstartMAG = tslice - timedelta(seconds=0.5)
        tendMAG = tslice + timedelta(seconds=0.5)
        tsliceindexStartMag = bisect.bisect_left(Mag_Epoch, tstartMAG)
        tsliceindexEndMag = bisect.bisect_right(Mag_Epoch, tendMAG)
        magF_SRF = B_SRF[tsliceindexStartMag:tsliceindexEndMag].mean(axis=0)

        PSDEAS1_DataSlice = EAS1_PSD[index]
        PSDEAS2_DataSlice = EAS2_PSD[index]
        PSDEAS1_CNTslice = EAS1_CNT[index]
        PSDEAS2_CNTslice = EAS2_CNT[index]

        # Remove the one-particle noise level.
        PSD_EAS1slice = np.copy(PSDEAS1_DataSlice)
        PSD_EAS1slice[PSDEAS1_CNTslice <= 1.0] = 0
        PSD_EAS2slice = np.copy(PSDEAS2_DataSlice)
        PSD_EAS2slice[PSDEAS2_CNTslice <= 1.0] = 0

        print('EAS1 Before Removing Noise: ', len(np.where(PSDEAS1_DataSlice > 0)[0]))
        print('EAS1 After Removing Noise: ', len(np.where(PSD_EAS1slice > 0)[0]))
        print('EAS2 Before Removing Noise: ', len(np.where(PSDEAS2_DataSlice > 0)[0]))
        print('EAS2 After Removing Noise: ', len(np.where(PSD_EAS2slice > 0)[0]))

        # Transfer to (Azi, Ele, Vel) sequence.
        PSD_EAS1slice = np.transpose(PSD_EAS1slice, (2, 0, 1))
        PSD_EAS2slice = np.transpose(PSD_EAS2slice, (2, 0, 1))
        PSD_EAS1slice = PSD_EAS1slice / 1e18  # from s^3 km^-6 to s^3 m^-6
        PSD_EAS2slice = PSD_EAS2slice / 1e18  # from s^3 km^-6 to s^3 m^-6

        vxEAS1, vyEAS1, vzEAS1 = GetVelocityVector(thetaEAS1, phiEAS1, velocityEAS1)
        vxEAS2, vyEAS2, vzEAS2 = GetVelocityVector(thetaEAS2, phiEAS2, velocityEAS2)

        vxEAS1_SRF, vyEAS1_SRF, vzEAS1_SRF = Transfer2SRF(EAS1_To_SRF, vxEAS1, vyEAS1, vzEAS1)
        vxEAS2_SRF, vyEAS2_SRF, vzEAS2_SRF = Transfer2SRF(EAS2_To_SRF, vxEAS2, vyEAS2, vzEAS2)

        # Transfer to PAS sequence.
        vxEAS1_SRF = np.transpose(vxEAS1_SRF, (2, 0, 1))
        vyEAS1_SRF = np.transpose(vyEAS1_SRF, (2, 0, 1))
        vzEAS1_SRF = np.transpose(vzEAS1_SRF, (2, 0, 1))
        vxEAS2_SRF = np.transpose(vxEAS2_SRF, (2, 0, 1))
        vyEAS2_SRF = np.transpose(vyEAS2_SRF, (2, 0, 1))
        vzEAS2_SRF = np.transpose(vzEAS2_SRF, (2, 0, 1))

        # Transfer the vectors to field-aligned coordinates.
        (Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz) = fieldAlignedCoordinates(magF_SRF[0], magF_SRF[1], magF_SRF[2])
        VparaEAS1, Vperp1EAS1, Vperp2EAS1 = rotateVectorIntoFieldAligned(vxEAS1_SRF, vyEAS1_SRF, vzEAS1_SRF, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
        VperpEAS1 = np.sqrt(Vperp1EAS1**2 + Vperp2EAS1**2)
        VparaEAS2, Vperp1EAS2, Vperp2EAS2 = rotateVectorIntoFieldAligned(vxEAS2_SRF, vyEAS2_SRF, vzEAS2_SRF, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
        VperpEAS2 = np.sqrt(Vperp1EAS2**2 + Vperp2EAS2**2)

        # Seperate energy ranges.
        LowEnergyMask = np.zeros(energyEAS1.shape, dtype=bool)
        CoreEnergyMask = np.zeros(energyEAS1.shape, dtype=bool)
        StrahlEnergyMask = np.zeros(energyEAS1.shape, dtype=bool)
        LowEnergyMask[np.where(energyEAS1 < 10.1)] = True
        CoreEnergyMask[np.where((energyEAS1 >= 10) & (energyEAS1 < 71))] = True
        StrahlEnergyMask[np.where(energyEAS1 >= 71)] = True

        VparaEAS1Low = VparaEAS1[:, :, LowEnergyMask]
        VperpEAS1Low = VperpEAS1[:, :, LowEnergyMask]
        PSDEAS1Low = PSD_EAS1slice[:, :, LowEnergyMask]
        VparaEAS1Core = VparaEAS1[:, :, CoreEnergyMask]
        VperpEAS1Core = VperpEAS1[:, :, CoreEnergyMask]
        PSDEAS1Core = PSD_EAS1slice[:, :, CoreEnergyMask]
        VparaEAS1Strahl = VparaEAS1[:, :, StrahlEnergyMask]
        VperpEAS1Strahl = VperpEAS1[:, :, StrahlEnergyMask]
        PSDEAS1Strahl = PSD_EAS1slice[:, :, StrahlEnergyMask]

        VparaEAS2Low = VparaEAS2[:, :, LowEnergyMask]
        VperpEAS2Low = VperpEAS2[:, :, LowEnergyMask]
        PSDEAS2Low = PSD_EAS2slice[:, :, LowEnergyMask]
        VparaEAS2Core = VparaEAS2[:, :, CoreEnergyMask]
        VperpEAS2Core = VperpEAS2[:, :, CoreEnergyMask]
        PSDEAS2Core = PSD_EAS2slice[:, :, CoreEnergyMask]
        VparaEAS2Strahl = VparaEAS2[:, :, StrahlEnergyMask]
        VperpEAS2Strahl = VperpEAS2[:, :, StrahlEnergyMask]
        PSDEAS2Strahl = PSD_EAS2slice[:, :, StrahlEnergyMask]

        # Interpolate to 2D for plotting.
        # So we can track every step!!!!
        VparaLowEAS1_2D, VperpLowEAS1_2D, VDFLowEAS1_2D = interpolate2D(VparaEAS1Low, VperpEAS1Low, PSD_EAS1slice[:, :, LowEnergyMask])
        VparaCoreEAS1_2D, VperpCoreEAS1_2D, VDFCoreEAS1_2D = interpolate2D(VparaEAS1Core, VperpEAS1Core, PSD_EAS1slice[:, :, CoreEnergyMask])
        VparaStrahlEAS1_2D, VperpStrahlEAS1_2D, VDFStrahlEAS1_2D = interpolate2D(VparaEAS1Strahl, VperpEAS1Strahl, PSD_EAS1slice[:, :, StrahlEnergyMask])
        VparaLowEAS2_2D, VperpLowEAS2_2D, VDFLowEAS2_2D = interpolate2D(VparaEAS2Low, VperpEAS2Low, PSD_EAS2slice[:, :, LowEnergyMask])
        VparaCoreEAS2_2D, VperpCoreEAS2_2D, VDFCoreEAS2_2D = interpolate2D(VparaEAS2Core, VperpEAS2Core, PSD_EAS2slice[:, :, CoreEnergyMask])
        VparaStrahlEAS2_2D, VperpStrahlEAS2_2D, VDFStrahlEAS2_2D = interpolate2D(VparaEAS2Strahl, VperpEAS2Strahl, PSD_EAS2slice[:, :, StrahlEnergyMask])

        # Plot to see the data before fitting.
        VDFmin = 1e-18
        VDFmax = max(PSD_EAS1slice.max(), PSD_EAS2slice.max())

        # See distribution.
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        pc0 = axes[0, 0].pcolormesh(VparaLowEAS1_2D, VperpLowEAS1_2D, VDFLowEAS1_2D.T, shading='auto', norm=colors.LogNorm(vmin=VDFmin, vmax=VDFmax))
        axes[0, 0].set_title('EAS1 Low Energy')
        axes[0, 0].set_ylabel('Vperp (m/s)')
        axes[0, 0].set_xlim(-2e7, 2e7)
        axes[0, 0].set_ylim(0, 2e7)

        pc1 = axes[0, 1].pcolormesh(VparaCoreEAS1_2D, VperpCoreEAS1_2D, VDFCoreEAS1_2D.T, shading='auto', norm=colors.LogNorm(vmin=VDFmin, vmax=VDFmax))
        axes[0, 1].set_title('EAS1 Core Energy')
        axes[0, 1].set_xlim(-2e7, 2e7)
        axes[0, 1].set_ylim(0, 2e7)

        pc2 = axes[0, 2].pcolormesh(VparaStrahlEAS1_2D, VperpStrahlEAS1_2D, VDFStrahlEAS1_2D.T, shading='auto', norm=colors.LogNorm(vmin=VDFmin, vmax=VDFmax))
        axes[0, 2].set_title('EAS1 Strahl Energy')
        axes[0, 2].set_xlim(-2e7, 2e7)
        axes[0, 2].set_ylim(0, 2e7)

        pc3 = axes[1, 0].pcolormesh(VparaLowEAS2_2D, VperpLowEAS2_2D, VDFLowEAS2_2D.T, shading='auto', norm=colors.LogNorm(vmin=VDFmin, vmax=VDFmax))
        axes[1, 0].set_title('EAS2 Low Energy')
        axes[1, 0].set_xlabel('Vpara (m/s)')
        axes[1, 0].set_ylabel('Vperp (m/s)')
        axes[1, 0].set_xlim(-2e7, 2e7)
        axes[1, 0].set_ylim(0, 2e7)

        pc4 = axes[1, 1].pcolormesh(VparaCoreEAS2_2D, VperpCoreEAS2_2D, VDFCoreEAS2_2D.T, shading='auto', norm=colors.LogNorm(vmin=VDFmin, vmax=VDFmax))
        axes[1, 1].set_title('EAS2 Core Energy')
        axes[1, 1].set_xlabel('Vpara (m/s)')
        axes[1, 1].set_xlim(-2e7, 2e7)
        axes[1, 1].set_ylim(0, 2e7)

        pc5 = axes[1, 2].pcolormesh(VparaStrahlEAS2_2D, VperpStrahlEAS2_2D, VDFStrahlEAS2_2D.T, shading='auto', norm=colors.LogNorm(vmin=VDFmin, vmax=VDFmax))
        axes[1, 2].set_title('EAS2 Strahl Energy')
        axes[1, 2].set_xlabel('Vpara (m/s)')
        axes[1, 2].set_xlim(-2e7, 2e7)
        axes[1, 2].set_ylim(0, 2e7)
        fig.colorbar(pc0, ax=axes)

        plt.savefig(SaveDir + 'PSD_Before_fitting.png')
        plt.close()

        # Use a bi-Maxwellian to fit the core range.
        # Non-zero points.
        # EAS1
        NonZeroCoreEAS1 = np.where(PSDEAS1Core > 0)
        NonZeroVparaCoreEAS1 = VparaEAS1Core[NonZeroCoreEAS1]
        NonZeroVperpCoreEAS1 = VperpEAS1Core[NonZeroCoreEAS1]
        NonZeroPSDCoreEAS1 = PSDEAS1Core[NonZeroCoreEAS1]
        # EAS2 
        NonZeroCoreEAS2 = np.where(PSDEAS2Core > 0)
        NonZeroVparaCoreEAS2 = VparaEAS2Core[NonZeroCoreEAS2]
        NonZeroVperpCoreEAS2 = VperpEAS2Core[NonZeroCoreEAS2]
        NonZeroPSDCoreEAS2 = PSDEAS2Core[NonZeroCoreEAS2]

        NonZeroVparaCore = np.concatenate((NonZeroVparaCoreEAS1, NonZeroVparaCoreEAS2))
        NonZeroVperpCore = np.concatenate((NonZeroVperpCoreEAS1, NonZeroVperpCoreEAS2))
        NonZeroPSDCore = np.concatenate((NonZeroPSDCoreEAS1, NonZeroPSDCoreEAS2))

        # Fit the bi-Maxwellian.
        params_fit = fit_bi_Maxwellian(NonZeroPSDCore, NonZeroVparaCore, NonZeroVperpCore)
        print(params_fit)

        # Write params_fit to a csvfile.
        csv_filename = SaveDir + 'params_fit.csv'
        with open(csv_filename, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Parameter', 'Value'])  # Write header
            for key, value in params_fit.items():
                csv_writer.writerow([key, value])  # Write each parameter and its value

        # Make a gird based on the fitting.
        Vpara_bins = np.linspace(-1.8e6, 1.8e6, 49)
        Vperp_bins = np.linspace(0.02, 1.8e6, 24)
        Vparabins_Grid, Vperpbins_Grid = np.meshgrid(Vpara_bins, Vperp_bins, indexing='ij')

        # Get the low-energy part.
        PSDLowEAS1_fit = bi_Maxwellian_2D(VparaEAS1Low, VperpEAS1Low, params_fit['n'], params_fit['Tpara'], params_fit['Tperp'], params_fit['v_drift'])
        PSDLowEAS2_fit = bi_Maxwellian_2D(VparaEAS2Low, VperpEAS2Low, params_fit['n'], params_fit['Tpara'], params_fit['Tperp'], params_fit['v_drift'])

        # Replace the low-energy range with fitted values.
        PSD_EAS1slice[:, :, LowEnergyMask] = PSDLowEAS1_fit
        PSD_EAS2slice[:, :, LowEnergyMask] = PSDLowEAS2_fit

        # Now, we can store everything.
        # In Spherical
        Electrons_EAS1_Spherical = SolarWindParticle('electron', time=tslice, magfield=magF_SRF, grid=[thetaEAS1, phiEAS1, velocityEAS1], coord_type='Spherical')
        Electrons_EAS2_Spherical = SolarWindParticle('electron', time=tslice, magfield=magF_SRF, grid=[thetaEAS2, phiEAS2, velocityEAS2], coord_type='Spherical')

        Electrons_EAS1_Spherical.set_vdf(PSD_EAS1slice)
        Electrons_EAS2_Spherical.set_vdf(PSD_EAS2slice)

        # In 3D Field-Aligned Coordinates.
        Electrons_EAS1_FieldAligned = SolarWindParticle('electron', time=tslice, magfield=magF_SRF, grid=[VparaEAS1, Vperp1EAS1, Vperp2EAS1], coord_type='3D Field-Aligned')
        Electrons_EAS2_FieldAligned = SolarWindParticle('electron', time=tslice, magfield=magF_SRF, grid=[VparaEAS2, Vperp1EAS2, Vperp2EAS2], coord_type='3D Field-Aligned')

        Electrons_EAS1_FieldAligned.set_vdf(PSD_EAS1slice)
        Electrons_EAS2_FieldAligned.set_vdf(PSD_EAS2slice)

        # In 2D Field-Aligned Coordinates.
        EAS1_2D = FieldAligned3D_To_2D(Electrons_EAS1_FieldAligned)
        EAS2_2D = FieldAligned3D_To_2D(Electrons_EAS2_FieldAligned)

        # save the pickle files.
        save_pickle(path=SaveDir + 'Electron_EAS1_Spherical.pkl', data=Electrons_EAS1_Spherical)
        save_pickle(path=SaveDir + 'Electron_EAS2_Spherical.pkl', data=Electrons_EAS2_Spherical)

        save_pickle(path=SaveDir + 'Electron_EAS1_FieldAligned.pkl', data=Electrons_EAS1_FieldAligned)
        save_pickle(path=SaveDir + 'Electron_EAS2_FieldAligned.pkl', data=Electrons_EAS2_FieldAligned)

        # Also, plot to see how the fitting works.
        VparaLowEAS1_2D_afterfitting, VperpLowEAS1_2D_afterfitting, VDFLowEAS1_2D_afterfitting = interpolate2D(VparaEAS1Low, VperpEAS1Low, PSDLowEAS1_fit)
        VparaLowEAS2_2D_afterfitting, VperpLowEAS2_2D_afterfitting, VDFLowEAS2_2D_afterfitting = interpolate2D(VparaEAS2Low, VperpEAS2Low, PSDLowEAS2_fit)

        VDFmax_EAS1 = max(VDFLowEAS1_2D.max(), np.max(EAS1_2D.get_vdf()))
        VDFmax_EAS2 = max(VDFLowEAS2_2D.max(), np.max(EAS2_2D.get_vdf()))
        VDFmin = 5e-14

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        pc0 = axes[0, 0].pcolormesh(Vparabins_Grid, Vperpbins_Grid, VDFLowEAS1_2D, shading='auto', norm=colors.LogNorm(vmin=VDFmin, vmax=VDFmax_EAS1))
        axes[0, 0].set_title('EAS1 Low Energy Before Fitting')
        axes[0, 0].set_ylabel('Vperp (m/s)')

        pc1 = axes[0, 1].pcolormesh(VparaLowEAS1_2D_afterfitting, VperpLowEAS1_2D_afterfitting, VDFLowEAS1_2D_afterfitting.T, shading='auto', norm=colors.LogNorm(vmin=VDFmin, vmax=VDFmax_EAS1))
        axes[0, 1].set_title('EAS1 Low Energy After Fitting')

        pc2 = axes[1, 0].pcolormesh(Vparabins_Grid, Vperpbins_Grid, VDFLowEAS2_2D, shading='auto', norm=colors.LogNorm(vmin=VDFmin, vmax=VDFmax_EAS2))
        axes[1, 0].set_title('EAS2 Low Energy Before Fitting')
        axes[1, 0].set_xlabel('Vpara (m/s)')
        axes[1, 0].set_ylabel('Vperp (m/s)')

        pc3 = axes[1, 1].pcolormesh(VparaLowEAS2_2D_afterfitting, VperpLowEAS2_2D_afterfitting, VDFLowEAS2_2D_afterfitting.T, shading='auto', norm=colors.LogNorm(vmin=VDFmin, vmax=VDFmax_EAS2))
        axes[1, 1].set_title('EAS2 Low Energy After Fitting')
        axes[1, 1].set_xlabel('Vpara (m/s)')

        plt.savefig(SaveDir + "After_Fitting.png")
        plt.close()


    return 0


if __name__ == '__main__':
    main()
