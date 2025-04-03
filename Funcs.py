"""
Author = @RanHao
Starting Date = 2024.10.09
In this Python script, I will define all the functions that I will use in the main scripts.
"""


import os, sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import scipy
import scipy.interpolate
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plasmapy
import math
import pickle

import scipy.constants as const
import astropy.units as u


class SolarWind:
    """
    The object stores the: Grid of measurements (V_para, V_perp1, V_perp2), and VDF of different species (vdf_proton, alpha, and electron).
    """

    def __init__(self, time, elevation, azimuth, velocity, magfield, dist_params=None,
                 vdf_proton=None, vdf_alpha=None, vdf_electron=None):
        self.time = time
        self.elevation = elevation.to(u.deg).value    # in deg
        self.azimuth = azimuth.to(u.deg).value        # in deg
        self.velocity = velocity.to(u.m / u.s).value  # in m /s
        self.magfield = magfield
        self.dist_params = dist_params                # Distribution parameters from GMM (means, covariances, weights), for error estimation, [means, covariances, weights]

        # Initialize VDFs
        self.vdf = {
            "proton": vdf_proton,  # Can be a single array or a dict {"core": ..., "beam": ...}
            "alpha": vdf_alpha,    # Same flexibility as proton
            "electron": np.array(vdf_electron) if vdf_electron is not None else None
        }

    def set_vdf(self, species, vdf, component=None):
        """
        Set a VDF for a species or its sub-component (core/beam).
        If no component is specified, store as a single VDF.
        """
        if species not in self.vdf:
            raise ValueError("Species not in the list. Use proton, alpha, or electron.")
        
        # Check for single or sub-component VDF
        if component is None:
            self.vdf[species] = np.array(vdf)
        else:
            if self.vdf[species] is None or not isinstance(self.vdf[species], dict):
                self.vdf[species] = {"core": None, "beam": None}
            if component not in self.vdf[species]:
                raise ValueError(f"Invalid component: {component}. Use 'core' or 'beam'.")
            # Explicitly check if vdf is None instead of relying on `if vdf`
            if vdf is None:
                raise ValueError(f"VDF data cannot be None for {species} {component}.")
            self.vdf[species][component] = np.array(vdf)

    def get_vdf(self, species, component=None):
        if species not in self.vdf:
            raise ValueError("Species not in the list. Use proton, alpha, or electron.")
        if component is None:
            return self.vdf[species]
        else:
            if not isinstance(self.vdf[species], dict) or component not in self.vdf[species]:
                raise ValueError(f"Component not found for {species}: {component}.")
            return self.vdf[species][component]

    def cal_density(self, species, component=None):
        if species not in self.vdf:
            raise ValueError("Species not in the list. Use proton, alpha, or electron.")
        vdf_data = None

        if isinstance(self.vdf[species], dict):
            if component:
                vdf_data = self.get_vdf(species, component)
            else:
                vdf_data = sum(self.vdf[species][comp] for comp in self.vdf[species] if self.vdf[species][comp] is not None)
        else:
            vdf_data = self.vdf[species]

        if vdf_data is None:
            raise ValueError(f"No VDF data available for {species} {component if component else ''}.")

        subazi, subele, subspeed = np.meshgrid(self.azimuth, self.elevation, self.velocity, indexing='ij')
        jacobian = subspeed**2 * np.cos(np.radians(subele))
        int_kernal = vdf_data * jacobian
        int_ele = np.trapz(int_kernal, x=np.radians(self.elevation), axis=1)
        int_ele_energy = -np.trapz(int_ele, x=self.velocity, axis=1)
        int_all = np.trapz(int_ele_energy, x=np.radians(self.azimuth), axis=0)

        return int_all

    def cal_bulk_velocity(self, species, component=None):
        if species not in self.vdf:
            raise ValueError("Species not in the list. Use proton, alpha, or electron.")
        vdf_data = None

        if isinstance(self.vdf[species], dict):
            if component:
                vdf_data = self.get_vdf(species, component)
            else:
                vdf_data = sum(self.vdf[species][comp] for comp in self.vdf[species] if self.vdf[species][comp] is not None)
        else:
            vdf_data = self.vdf[species]

        if vdf_data is None:
            raise ValueError(f"No VDF data available for {species} {component if component else ''}.")

        subazi, subele, subspeed = np.meshgrid(self.azimuth, self.elevation, self.velocity, indexing='ij')

        jacobian = subspeed**2 * np.cos(np.radians(subele))
        vx_kernal = -vdf_data * subspeed * np.cos(np.radians(subele)) * np.cos(np.radians(subazi)) * jacobian
        vy_kernal = vdf_data * subspeed * np.cos(np.radians(subele)) * np.sin(np.radians(subazi)) * jacobian
        vz_kernal = -vdf_data * subspeed * np.sin(np.radians(subele)) * jacobian

        vx_ele = np.trapz(vx_kernal, x=np.radians(self.elevation), axis=1)
        vy_ele = np.trapz(vy_kernal, x=np.radians(self.elevation), axis=1)
        vz_ele = np.trapz(vz_kernal, x=np.radians(self.elevation), axis=1)

        vx_ele_energy = -np.trapz(vx_ele, x=self.velocity, axis=1)
        vy_ele_energy = -np.trapz(vy_ele, x=self.velocity, axis=1)
        vz_ele_energy = -np.trapz(vz_ele, x=self.velocity, axis=1)

        vx_all = np.trapz(vx_ele_energy, x=np.radians(self.azimuth), axis=0)
        vy_all = np.trapz(vy_ele_energy, x=np.radians(self.azimuth), axis=0)
        vz_all = np.trapz(vz_ele_energy, x=np.radians(self.azimuth), axis=0)

        density = self.cal_density(species, component)
        vx_bulk = vx_all / density
        vy_bulk = vy_all / density
        vz_bulk = vz_all / density

        return np.array([vx_bulk, vy_bulk, vz_bulk])

    def cal_pressure_tensor(self, species, component=None):
        if species not in self.vdf:
            raise ValueError("Species not in the list. Use proton, alpha, or electron.")
        vdf_data = None

        if isinstance(self.vdf[species], dict):
            if component:
                vdf_data = self.get_vdf(species, component)
            else:
                vdf_data = sum(self.vdf[species][comp] for comp in self.vdf[species] if self.vdf[species][comp] is not None)
        else:
            vdf_data = self.vdf[species]

        if vdf_data is None:
            raise ValueError(f"No VDF data available for {species} {component if component else ''}.")

        mass_dict = {"proton": 1.6726219e-27, "alpha": 6.64424e-27, "electron": 9.10938356e-31}
        mass = mass_dict.get(species)

        vx_bulk, vy_bulk, vz_bulk = self.cal_bulk_velocity(species, component)

        subazi, subele, subspeed = np.meshgrid(self.azimuth, self.elevation, self.velocity, indexing='ij')
        vx = -subspeed * np.cos(np.radians(subele)) * np.cos(np.radians(subazi))
        vy = subspeed * np.cos(np.radians(subele)) * np.sin(np.radians(subazi))
        vz = -subspeed * np.sin(np.radians(subele))

        dvx = vx - vx_bulk
        dvy = vy - vy_bulk
        dvz = vz - vz_bulk

        jacobian = subspeed**2 * np.cos(np.radians(subele))
        Pxx_kernal = vdf_data * mass * dvx * dvx * jacobian
        Pxy_kernal = vdf_data * mass * dvx * dvy * jacobian
        Pxz_kernal = vdf_data * mass * dvx * dvz * jacobian
        Pyy_kernal = vdf_data * mass * dvy * dvy * jacobian
        Pyz_kernal = vdf_data * mass * dvy * dvz * jacobian
        Pzz_kernal = vdf_data * mass * dvz * dvz * jacobian

        def integrate_pressure(kernal):
            int_ele = np.trapz(kernal, x=np.radians(self.elevation), axis=1)
            int_energy = -np.trapz(int_ele, x=self.velocity, axis=1)
            return np.trapz(int_energy, x=np.radians(self.azimuth), axis=0)

        Pxx = integrate_pressure(Pxx_kernal)
        Pxy = integrate_pressure(Pxy_kernal)
        Pxz = integrate_pressure(Pxz_kernal)
        Pyy = integrate_pressure(Pyy_kernal)
        Pyz = integrate_pressure(Pyz_kernal)
        Pzz = integrate_pressure(Pzz_kernal)

        pressure_tensor = np.array([[Pxx, Pxy, Pxz],
                                    [Pxy, Pyy, Pyz],
                                    [Pxz, Pyz, Pzz]])

        return pressure_tensor

    def __repr__(self):
        """
        Provide a summary of the SolarWind object, showing time, magnetic field,
        and the status of VDFs (set/not set for proton, alpha, and electron).
        """
        def get_status(vdf):
            if vdf is None:
                return "not set"
            if isinstance(vdf, dict):
                core_set = "set" if vdf.get("core") is not None else "not set"
                beam_set = "set" if vdf.get("beam") is not None else "not set"
                return f"core: {core_set}, beam: {beam_set}"
            return "single VDF set"

        proton_status = get_status(self.vdf["proton"])
        alpha_status = get_status(self.vdf["alpha"])
        electron_status = "set" if self.vdf["electron"] is not None else "not set"
        
        return (f"Time = {self.time}, "
                f"Magnetic Field = {self.magfield}, "
                f"Proton VDF = {proton_status}, "
                f"Alpha VDF = {alpha_status}, "
                f"Electron VDF = {electron_status}")


def resample_to_target_time(dataframe, target_tstart, target_tend, target_dt, keys, units=None):
    # The function for resampling the data to the target time resolution.
    # Turn the index of the dataframe to numbers according to their gap to the first point (time index).
    time = dataframe.index
    time_index_in_num = [(time[i] - time[0]).total_seconds() for i in range(len(time))]

    # Derive the time index numbers of the target time range.
    target_time_gap = (target_tend - target_tstart).total_seconds()
    point_num = int(target_time_gap / target_dt)
    target_start_index = (target_tstart - time[0]).total_seconds()
    target_end_index = (target_tend - time[0]).total_seconds()

    target_timeindex_in_num = np.linspace(target_start_index, target_end_index, point_num + 1)


    res_dict = {}
    for i in range(len(keys)):
        values = dataframe[keys[i]].values
        f = scipy.interpolate.interp1d(time_index_in_num, values)
        y_new = f(target_timeindex_in_num)
        res_dict[keys[i]] = y_new

    target_timestamps = [time[0] + timedelta(seconds=x) for x in target_timeindex_in_num]
    res_df = pd.DataFrame(res_dict, index=target_timestamps)

    return res_df

def butter_lowpass_filter(data, cutoff, fs, order=5):
    # The butter low pass filter
    def butter_lowpass(cutoff, fs, order=5):
        return butter(order, cutoff, fs=fs, btype='low')

    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def magnitude(vector):
    # calculate the magnitude of a vector
    return math.sqrt(sum(pow(element, 2) for element in vector))


def fieldAlignedCoordinates(Bx, By, Bz):
    '''
    INPUTS:
         Bx, By, Bz = rank1 arrays of magnetic field measurements in SRF frame
    '''
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

    # Define field-aligned vector
    Nx = Bx/Bmag
    Ny = By/Bmag
    Nz = Bz/Bmag

    # Make up some unit vector
    if np.isscalar(Nx):
        Rx = 0
        Ry = 1.
        Rz = 0
    else:
        Rx = np.zeros(Nx.len())
        Ry = np.ones(len(Nx))
        Rz = np.zeros(len(Nx))

    # Find some vector perpendicular to field NxR 
    TEMP_Px = ( Ny * Rz ) - ( Nz * Ry )  # P = NxR
    TEMP_Py = ( Nz * Rx ) - ( Nx * Rz )  # This is temporary in case we choose a vector R that is not unitary
    TEMP_Pz = ( Nx * Ry ) - ( Ny * Rx )


    Pmag = np.sqrt( TEMP_Px**2 + TEMP_Py**2 + TEMP_Pz**2 ) #Have to normalize, since previous definition does not imply unitarity, just orthogonality
  
    Px = TEMP_Px / Pmag # for R=(0,1,0), NxR = P ~= RTN_N
    Py = TEMP_Py / Pmag
    Pz = TEMP_Pz / Pmag


    Qx = ( Pz * Ny ) - ( Py * Nz )   # N x P
    Qy = ( Px * Nz ) - ( Pz * Nx )  
    Qz = ( Py * Nx ) - ( Px * Ny )  

    return(Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)


def rotateVectorIntoFieldAligned(Ax, Ay, Az, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz):
    # For some Vector A in the SAME COORDINATE SYSTEM AS THE ORIGINAL B-FIELD VECTOR:

    An = (Ax * Nx) + (Ay * Ny) + (Az * Nz)  # A dot N = A_parallel
    Ap = (Ax * Px) + (Ay * Py) + (Az * Pz)  # A dot P = A_perp (~RTN_N (+/- depending on B), perpendicular to s/c y)
    Aq = (Ax * Qx) + (Ay * Qy) + (Az * Qz)  # 

    return(An, Ap, Aq)

def low_pass_filter(data, cutoff_frequency, sampling_rate):
    """
    Apply a simple low-pass filter to a 1D array of data.

    Parameters:
        data (list or numpy array): The input data to filter.
        cutoff_frequency (float): The cutoff frequency of the filter in Hz.
        sampling_rate (float): The sampling rate of the data in Hz.

    Returns:
        numpy array: The filtered data.
    """
    import numpy as np
    
    # Calculate the RC time constant
    dt = 1 / sampling_rate
    alpha = dt / (dt + (1 / (2 * np.pi * cutoff_frequency)))
    
    # Apply the low-pass filter
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]  # Initialize with the first value
    for i in range(1, len(data)):
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i - 1]
    
    return filtered_data

def PTensor2T(PTensor, Density, B_field, k2eV):
    k_B = 1.38064852e-23
    TTensor = PTensor / k_B / Density 

    B_magnitude = np.sqrt(np.sum(B_field**2))
    b_hat = B_field / B_magnitude

    # T_para
    T_para = np.dot(b_hat, np.dot(TTensor, b_hat))
    T_trace = np.trace(TTensor)
    T_perp = (T_trace - T_para) / 2 # K
    # K to eV
    T_para = T_para * k2eV
    T_perp = T_perp * k2eV
    
    return T_para, T_perp

def read_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    
def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def OneParticleNoiseLevel(count_cdffile, vdffile):

    # For all the measurement of the day
    vdf = vdffile['vdf'][...] # Extract all the vdf
    vdf_epochs = np.array(vdffile['Epoch'], dtype="datetime64[ms]") # Extract all the epochs, convert to np.array for quicker search.
    counts = count_cdffile['COUNTS'][...] # Extract all the counts
    count_epochs = np.array(count_cdffile['Epoch'], dtype="datetime64[ms]") # Extract all the epochs

    adjusted_vdf_times = vdf_epochs - np.timedelta64(500, 'ms') # Adjust the vdf times to match the counts times
    match_indices = np.searchsorted(count_epochs, adjusted_vdf_times) # Find the indices of the counts that match the vdf times
    counts = counts[match_indices] # Get the counts that match the vdf times

    noise_level = np.divide(vdf, counts, out=np.zeros_like(vdf, dtype=float), where=counts != 0) # Calculate the noise level
    noise_level[np.isnan(noise_level)] = 0 # Drop the NaN values and replace them with 0.
    one_particle_noise_level = np.nanpercentile(noise_level, 99.99, axis=0)

    return one_particle_noise_level
